def queue_based_batch():
    st.title("🚗 Car Damage Detection - Queue System")
    
    # Initialize session state for queue
    if 'car_queue' not in st.session_state:
        st.session_state.car_queue = []
    
    # === ADD CAR SECTION ===
    st.header("➕ Tambah Mobil")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        plate_number = st.text_input(
            "Plat Mobil:",
            placeholder="Contoh: B1234CD",
            help="Masukkan nomor plat untuk identifikasi"
        )
    
    with col2:
        car_photos = st.file_uploader(
            "Upload foto mobil ini:",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload semua foto untuk mobil ini sekaligus"
        )
    
    # Preview uploaded photos
    if car_photos:
        st.write("📸 Preview foto yang akan ditambahkan:")
        preview_cols = st.columns(min(len(car_photos), 4))
        for i, photo in enumerate(car_photos[:4]):
            with preview_cols[i]:
                img = Image.open(photo)
                st.image(img, caption=photo.name, use_column_width=True)
        
        if len(car_photos) > 4:
            st.info(f"... dan {len(car_photos)-4} foto lainnya")
    
    # Add to queue button
    if st.button("➕ Add to Queue", disabled=not (plate_number and car_photos)):
        # Add car to queue
        car_data = {
            'plate': plate_number.upper(),
            'photos': car_photos.copy(),  # Store the files
            'photo_count': len(car_photos),
            'added_time': datetime.now().strftime("%H:%M:%S")
        }
        
        st.session_state.car_queue.append(car_data)
        st.success(f"✅ {plate_number} ditambahkan ke queue!")
        st.rerun()  # Refresh to clear form
    
    # === QUEUE MANAGEMENT ===
    st.header("📋 Queue Mobil")
    
    if not st.session_state.car_queue:
        st.info("Queue kosong. Tambahkan mobil di atas untuk mulai.")
    else:
        st.write(f"**{len(st.session_state.car_queue)} mobil dalam queue:**")
        
        # Show queue items
        for i, car in enumerate(st.session_state.car_queue):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.write(f"**{i+1}. {car['plate']}**")
                st.caption(f"Added: {car['added_time']}")
            
            with col2:
                st.write(f"📸 {car['photo_count']} foto")
                # Show photo names
                photo_names = [f.name for f in car['photos']][:3]
                if len(car['photos']) > 3:
                    photo_names.append(f"... +{len(car['photos'])-3} more")
                st.caption(" • ".join(photo_names))
            
            with col3:
                if st.button("❌", key=f"remove_{i}", help="Remove from queue"):
                    st.session_state.car_queue.pop(i)
                    st.rerun()
        
        # Queue actions
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Process All Cars", type="primary"):
                st.session_state.processing = True
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Queue"):
                st.session_state.car_queue = []
                st.rerun()
        
        with col3:
            total_photos = sum(car['photo_count'] for car in st.session_state.car_queue)
            st.metric("Total Foto", total_photos)
    
    # === PROCESSING ===
    if st.session_state.get('processing', False):
        process_queue()

def process_queue():
    st.header("🚀 Processing Queue...")
    
    model = load_model()
    if not model:
        return
    
    results = {}
    
    # Progress tracking
    total_cars = len(st.session_state.car_queue)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, car_data in enumerate(st.session_state.car_queue):
        plate = car_data['plate']
        status_text.text(f"Processing {plate}... ({i+1}/{total_cars})")
        
        # Process all photos for this car
        car_damages = {damage: 0 for damage in DAMAGE_CLASSES.values()}
        processed_images = []
        
        for photo in car_data['photos']:
            try:
                image = Image.open(photo)
                annotated_img, damage_counts, severity = process_single_image(model, image, plate)
                
                # Aggregate damages
                for damage, count in damage_counts.items():
                    car_damages[damage] += count
                
                processed_images.append(annotated_img)
                
            except Exception as e:
                st.warning(f"Error processing {photo.name}: {e}")
        
        # Calculate overall severity for this car
        overall_severity = get_damage_severity(car_damages)
        
        results[plate] = {
            'damage_counts': car_damages,
            'severity': overall_severity,
            'processed_images': processed_images,
            'photo_count': len(processed_images)
        }
        
        progress_bar.progress((i + 1) / total_cars)
    
    status_text.text("✅ All cars processed!")
    
    # Display results
    display_batch_results(results)
    
    # Clear processing flag
    st.session_state.processing = False

def display_batch_results(results):
    st.header("📊 Batch Results")
    
    # Summary metrics
    total_cars = len(results)
    total_damages = sum(sum(data['damage_counts'].values()) for data in results.values())
    good_cars = sum(1 for data in results.values() if data['severity'] in ['Baik', 'Ringan'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Mobil", total_cars)
    with col2:
        st.metric("Total Kerusakan", total_damages)  
    with col3:
        st.metric("Kondisi Baik/Ringan", f"{good_cars}/{total_cars}")
    
    # Results table
    st.subheader("📋 Detail Results")
    
    df_data = []
    for plate, data in results.items():
        row = {"Plat_Mobil": plate, **data['damage_counts'], "Status": data['severity']}
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)
    
    # Download CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        "📄 Download CSV Report",
        data=csv_buffer.getvalue(),
        file_name=f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Sample images
    st.subheader("🖼️ Sample Results")
    sample_plates = list(results.keys())[:3]
    
    cols = st.columns(len(sample_plates))
    for i, plate in enumerate(sample_plates):
        with cols[i]:
            st.write(f"**{plate}**")
            st.write(f"Status: {results[plate]['severity']}")
            if results[plate]['processed_images']:
                st.image(results[plate]['processed_images'][0], use_column_width=True)