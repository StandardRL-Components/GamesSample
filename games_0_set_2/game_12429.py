import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:32:09.131023
# Source Brief: brief_02429.md
# Brief Index: 2429
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a physics-based vehicle platformer.
    The player races across a procedurally generated planet, collecting musical notes
    to fuel sonic defenses against waves of energy creatures.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    game_description = (
        "Race a physics-based vehicle across a procedurally generated planet, collecting musical notes "
        "to fuel sonic defenses against waves of energy creatures."
    )
    user_guide = (
        "Controls: Use ↑↓ to thrust forward/reverse and ←→ to tilt. "
        "Press space to fire a sonic blast."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 5, 20)
    COLOR_TERRAIN = (80, 70, 20)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_NOTE = (150, 255, 150)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_BLAST = (100, 150, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_PARTICLE_TRAIL = (100, 255, 100)
    COLOR_PARTICLE_EXPLOSION = (255, 150, 50)

    # Physics & Game Rules
    GRAVITY = 0.3
    THRUST_POWER = 0.6
    REVERSE_POWER = 0.2
    TILT_SPEED = 0.05
    FRICTION = 0.99
    MAX_STEPS = 5000
    NOTE_COLLECT_RADIUS = 20
    ENEMY_CONTACT_RADIUS = 15
    
    # Sonic Blast
    BLAST_COST = 10
    BLAST_GROW_SPEED = 10
    
    # Wave Mechanics
    WAVE_DURATION = 600 # steps
    
    # Procedural Generation
    TERRAIN_POINTS = 1000
    TERRAIN_AMPLITUDE = 70
    TERRAIN_FREQUENCY = 0.02
    TERRAIN_OCTAVES = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        
        if self.render_mode == "human":
            pygame.display.set_caption("Sonic Racer")
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        else:
            self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        
        self.clock = pygame.time.Clock()
        
        # Initialize state variables to None, they will be set in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.note_count = None
        self.wave_num = None
        self.wave_timer = None
        self.last_dist_milestone = None
        self.prev_space_held = None
        
        self.terrain_y = None
        self.camera_x = None
        
        self.notes = None
        self.enemies = None
        self.blasts = None
        self.particles = None
        
        self.vehicle_level = None
        self.blast_level = None
        
        # This call will also perform the first reset()
        # self.validate_implementation() # Removed for submission


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([100.0, 150.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_angle = 0.0
        
        self.note_count = 20
        self.wave_num = 0
        self.wave_timer = self.WAVE_DURATION
        self.last_dist_milestone = 0
        self.prev_space_held = False
        
        self.camera_x = 0.0
        
        self.notes = []
        self.enemies = []
        self.blasts = []
        self.particles = []
        
        self.vehicle_level = 1
        self.blast_level = 1

        self._generate_initial_terrain()
        self._spawn_initial_notes()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self.reset()

        reward = 0
        self.steps += 1
        
        # 1. Handle Input
        self._handle_input(action)
        
        # 2. Update Physics & Player
        self._update_player_physics()
        
        # 3. Update Camera
        target_camera_x = self.player_pos[0] - self.SCREEN_WIDTH / 4
        self.camera_x = 0.9 * self.camera_x + 0.1 * target_camera_x # Smooth camera
        
        # 4. Update Game Logic (waves, spawning)
        wave_survived = self._update_waves()
        if wave_survived:
            reward += 1.0 # Wave survival reward
            
        # 5. Update and collide entities
        self._update_entities()
        note_collected, player_hit = self._handle_collisions()
        
        if note_collected:
            reward += 0.1 # Note collection reward
        
        # Reward for forward movement
        reward += self.player_vel[0] * 0.01

        # Distance milestone reward
        dist_traveled = int(self.player_pos[0])
        if dist_traveled > self.last_dist_milestone + 1000:
            self.last_dist_milestone += 1000
            reward += 10.0
            # SFX: Milestone reached!
        
        self.score = dist_traveled

        # 6. Check Termination
        terminated = False
        if player_hit:
            terminated = True
            reward = -100.0 # Collision penalty
            self._create_explosion(self.player_pos, 50)
            # SFX: Player explosion
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward += 10.0 # Max steps bonus
        self.game_over = terminated
        
        # 7. Render and return
        obs = self._get_observation()
        if self.render_mode == "human":
            self._render_human()
            self.clock.tick(self.metadata["render_fps"])

        return obs, reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Vehicle tilt
        if movement == 3: # Left
            self.player_angle -= self.TILT_SPEED
        if movement == 4: # Right
            self.player_angle += self.TILT_SPEED
            
        # Vehicle thrust
        thrust_vec = np.array([math.cos(self.player_angle), math.sin(self.player_angle)])
        if movement == 1: # Up (forward)
            self.player_vel += thrust_vec * self.THRUST_POWER
        if movement == 2: # Down (reverse)
            self.player_vel -= thrust_vec * self.REVERSE_POWER
            
        # Sonic blast (on button press)
        if space_held and not self.prev_space_held and self.note_count >= self.BLAST_COST:
            self.note_count -= self.BLAST_COST
            blast_radius = 80 + 20 * (self.blast_level - 1)
            self.blasts.append({'pos': self.player_pos.copy(), 'radius': 10, 'max_radius': blast_radius})
            # SFX: Sonic Blast Deploy
        self.prev_space_held = space_held

    def _update_player_physics(self):
        # Apply gravity
        self.player_vel[1] += self.GRAVITY
        
        # Apply velocity and friction
        self.player_pos += self.player_vel
        self.player_vel *= self.FRICTION
        
        # Terrain interaction
        ground_y, ground_angle = self._get_terrain_height_and_slope(self.player_pos[0])
        
        if self.player_pos[1] > ground_y - 8: # Vehicle has some thickness
            self.player_pos[1] = ground_y - 8
            
            # Align to ground and apply bounce/friction
            vel_mag = np.linalg.norm(self.player_vel)
            self.player_vel[1] = -self.player_vel[1] * 0.3 # Small bounce
            self.player_vel[0] *= 0.9 # Ground friction
            
            # Smoothly align angle to terrain
            self.player_angle = self.player_angle * 0.9 + ground_angle * 0.1

        # Keep player from going off left of screen
        self.player_pos[0] = max(self.player_pos[0], self.camera_x + 10)

        # Particle trail
        if np.linalg.norm(self.player_vel) > 1.0:
            if self.steps % 3 == 0:
                self.particles.append(self._create_particle(self.player_pos, self.COLOR_PARTICLE_TRAIL, 20))

    def _update_waves(self):
        self.wave_timer -= 1
        if self.wave_timer <= 0:
            self.wave_num += 1
            self.wave_timer = self.WAVE_DURATION
            self._spawn_enemy_wave()
            
            # Check for upgrades
            if self.wave_num % 5 == 0:
                self.vehicle_level += 1
                self.blast_level += 1
            
            return True # Wave survived
        return False
        
    def _spawn_enemy_wave(self):
        num_enemies = 3 + self.wave_num
        speed = 1.0 + self.wave_num * 0.1
        
        for _ in range(num_enemies):
            side = random.choice(['top', 'right'])
            if side == 'top':
                x = self.camera_x + random.uniform(0, self.SCREEN_WIDTH)
                y = self.camera_x - 50
            else: # right
                x = self.camera_x + self.SCREEN_WIDTH + 50
                y = self.camera_x + random.uniform(0, self.SCREEN_HEIGHT)
            
            pos = np.array([x, y])
            angle = math.atan2(self.player_pos[1] - pos[1], self.player_pos[0] - pos[0])
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.enemies.append({'pos': pos, 'vel': vel})
            # SFX: Enemy spawn alert

    def _update_entities(self):
        # Update enemies
        for enemy in self.enemies:
            # Simple homing behavior
            target_vec = self.player_pos - enemy['pos']
            target_vec /= np.linalg.norm(target_vec) + 1e-6
            enemy['vel'] = enemy['vel'] * 0.95 + target_vec * 0.1 * (1.0 + self.wave_num * 0.1)
            enemy['pos'] += enemy['vel']

        # Update blasts
        for blast in self.blasts:
            blast['radius'] += self.BLAST_GROW_SPEED

        # Update particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            
        # Cleanup dead entities
        self.particles = [p for p in self.particles if p['life'] > 0]
        self.blasts = [b for b in self.blasts if b['radius'] < b['max_radius']]
        
        # Cleanup off-screen enemies and notes
        screen_left = self.camera_x - 100
        screen_right = self.camera_x + self.SCREEN_WIDTH + 100
        self.enemies = [e for e in self.enemies if screen_left < e['pos'][0] < screen_right]
        self.notes = [n for n in self.notes if screen_left < n[0] < screen_right]

    def _handle_collisions(self):
        note_collected = False
        player_hit = False
        
        # Player -> Note
        for note_pos in self.notes[:]:
            if np.linalg.norm(self.player_pos - note_pos) < self.NOTE_COLLECT_RADIUS:
                self.notes.remove(note_pos)
                self.note_count += 1
                note_collected = True
                # SFX: Note collect
                
        # Player -> Enemy
        for enemy in self.enemies:
            if np.linalg.norm(self.player_pos - enemy['pos']) < self.ENEMY_CONTACT_RADIUS:
                player_hit = True
                # Damage is terminal, handled in step()
        
        # Blast -> Enemy
        for blast in self.blasts:
            for enemy in self.enemies[:]:
                if np.linalg.norm(blast['pos'] - enemy['pos']) < blast['radius']:
                    self.enemies.remove(enemy)
                    self._create_explosion(enemy['pos'], 20)
                    # SFX: Enemy destroyed
                    
        return note_collected, player_hit

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_human(self):
        pygame.display.flip()

    def _render_game(self):
        # Draw terrain
        terrain_points_on_screen = []
        start_idx = max(0, int(self.camera_x))
        end_idx = min(len(self.terrain_y) - 1, int(self.camera_x + self.SCREEN_WIDTH) + 2)
        
        for i in range(start_idx, end_idx):
            terrain_points_on_screen.append((i - self.camera_x, self.terrain_y[i]))
            
        if len(terrain_points_on_screen) > 2:
            screen_poly = terrain_points_on_screen + [(self.SCREEN_WIDTH, self.SCREEN_HEIGHT), (0, self.SCREEN_HEIGHT)]
            pygame.draw.polygon(self.screen, self.COLOR_TERRAIN, screen_poly)

        # Draw notes
        for note_pos in self.notes:
            sx, sy = int(note_pos[0] - self.camera_x), int(note_pos[1])
            if 0 < sx < self.SCREEN_WIDTH and 0 < sy < self.SCREEN_HEIGHT:
                self._draw_glow_circle(self.screen, self.COLOR_NOTE, (sx, sy), 5, 10)

        # Draw enemies
        for enemy in self.enemies:
            sx, sy = int(enemy['pos'][0] - self.camera_x), int(enemy['pos'][1])
            if 0 < sx < self.SCREEN_WIDTH and 0 < sy < self.SCREEN_HEIGHT:
                self._draw_glow_circle(self.screen, self.COLOR_ENEMY, (sx, sy), 8, 15)
        
        # Draw particles
        for p in self.particles:
            sx, sy = int(p['pos'][0] - self.camera_x), int(p['pos'][1])
            radius = int(p['life'] / p['max_life'] * p['radius'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, radius, p['color'] + (150,))
                
        # Draw player
        if not self.game_over:
            self._draw_vehicle()

        # Draw blasts
        for blast in self.blasts:
            sx, sy = int(blast['pos'][0] - self.camera_x), int(blast['pos'][1])
            alpha = max(0, 255 * (1 - blast['radius'] / blast['max_radius']))
            pygame.gfxdraw.aacircle(self.screen, sx, sy, int(blast['radius']), self.COLOR_BLAST + (int(alpha),))
            pygame.gfxdraw.aacircle(self.screen, sx, sy, int(blast['radius'])-1, self.COLOR_BLAST + (int(alpha),))

    def _draw_vehicle(self):
        sx, sy = self.player_pos[0] - self.camera_x, self.player_pos[1]
        
        # Vehicle body as a rotated polygon
        size = 12
        v_shape = [
            np.array([size, 0]),
            np.array([-size, -size/2]),
            np.array([-size, size/2])
        ]
        
        rotation = np.array([
            [math.cos(self.player_angle), -math.sin(self.player_angle)],
            [math.sin(self.player_angle), math.cos(self.player_angle)]
        ])
        
        points = [v_shape[i].dot(rotation) + (sx, sy) for i in range(3)]
        int_points = [(int(p[0]), int(p[1])) for p in points]
        
        pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_PLAYER)

    def _render_ui(self):
        dist_text = self.font_ui.render(f"DIST: {self.score:05d}m", True, self.COLOR_UI_TEXT)
        wave_text = self.font_ui.render(f"WAVE: {self.wave_num}", True, self.COLOR_UI_TEXT)
        note_text = self.font_ui.render(f"NOTES: {self.note_count}", True, self.COLOR_NOTE)
        
        self.screen.blit(dist_text, (10, 10))
        self.screen.blit(wave_text, (10, 30))
        self.screen.blit(note_text, (10, 50))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_num,
            "notes": self.note_count,
            "player_x": self.player_pos[0],
            "player_y": self.player_pos[1],
        }
        
    def _generate_initial_terrain(self):
        self.terrain_y = [0] * self.TERRAIN_POINTS
        y = self.SCREEN_HEIGHT / 2
        for i in range(self.TERRAIN_POINTS):
            y_offset = 0
            for o in range(self.TERRAIN_OCTAVES):
                freq = self.TERRAIN_FREQUENCY * (2**o)
                amp = self.TERRAIN_AMPLITUDE / (2**o)
                y_offset += math.sin(i * freq) * amp
            self.terrain_y[i] = int(self.SCREEN_HEIGHT * 0.75 + y_offset)
            
    def _get_terrain_height_and_slope(self, x_pos):
        x_idx = int(x_pos)
        if not (0 < x_idx < len(self.terrain_y) - 1):
            return self.SCREEN_HEIGHT, 0.0 # Out of bounds
        
        # Linear interpolation for height
        y1 = self.terrain_y[x_idx]
        y2 = self.terrain_y[x_idx + 1]
        height = y1 + (y2 - y1) * (x_pos - x_idx)
        
        # Slope
        slope_angle = math.atan2(y2 - y1, 1)
        
        return height, slope_angle

    def _spawn_initial_notes(self):
        for i in range(20):
            x = random.uniform(200, self.TERRAIN_POINTS - 200)
            ground_y, _ = self._get_terrain_height_and_slope(x)
            y = ground_y - random.uniform(30, 100)
            self.notes.append(np.array([x, y]))
            
    def _draw_glow_circle(self, surface, color, center, radius, glow_width):
        for i in range(glow_width, 0, -1):
            alpha = int(100 * (1 - i / glow_width))
            pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), radius + i, color + (alpha,))
        pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), radius, color)
        pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), radius, color)
        
    def _create_particle(self, pos, color, lifetime):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0.5, 2)
        vel = np.array([math.cos(angle), math.sin(angle)]) * speed
        return {
            'pos': pos.copy(), 'vel': vel, 'radius': random.randint(2, 4),
            'color': color, 'life': lifetime, 'max_life': lifetime
        }
        
    def _create_explosion(self, pos, num_particles):
        for _ in range(num_particles):
            p = self._create_particle(pos, self.COLOR_PARTICLE_EXPLOSION, random.randint(30, 60))
            p['vel'] *= random.uniform(2, 5) # Make explosion particles faster
            self.particles.append(p)
            
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset and observation space
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# --- Main block for human play testing ---
if __name__ == '__main__':
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    terminated = False
    
    # Key mapping for human control
    key_map = {
        pygame.K_w: 1, pygame.K_UP: 1,
        pygame.K_s: 2, pygame.K_DOWN: 2,
        pygame.K_a: 3, pygame.K_LEFT: 3,
        pygame.K_d: 4, pygame.K_RIGHT: 4,
    }
    
    movement = 0
    space_held = 0
    shift_held = 0
    
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key in key_map:
                    movement = key_map[event.key]
                if event.key == pygame.K_SPACE:
                    space_held = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 1
            if event.type == pygame.KEYUP:
                if event.key in key_map and movement == key_map[event.key]:
                    movement = 0
                if event.key == pygame.K_SPACE:
                    space_held = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 0

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Info: {info}")
            # Optional: Short pause before reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

    env.close()