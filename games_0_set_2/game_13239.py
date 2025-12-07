import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T20:00:16.691380
# Source Brief: brief_03239.md
# Brief Index: 3239
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A high-speed rhythm racer where you flip gravity to navigate between two-tiered tracks. "
        "Collect instruments and ride colored tiles to boost your score before time runs out."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to steer. Press space to flip gravity between the top and bottom tracks. "
        "Press shift to cycle through unlocked vehicle skins."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    
    # Colors
    COLOR_BG_TOP = (10, 5, 30)
    COLOR_BG_BOTTOM = (40, 10, 60)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (255, 255, 0, 50)
    
    TILE_COLORS = {
        "green": (50, 255, 150),
        "red": (255, 100, 100),
        "blue": (100, 150, 255)
    }
    
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_INSTRUMENT = (255, 180, 255)
    COLOR_FINISH_LINE = (255, 255, 255)

    # Physics & Gameplay
    GRAVITY = 0.8
    PLAYER_STEER_FORCE = 1.2
    PLAYER_FLIP_KICK = 10
    PLAYER_MAX_SPEED_X = 12
    PLAYER_BASE_SPEED_X = 4.0
    PLAYER_DRAG = 0.95
    
    TRACK_LENGTH_TILES = 300
    TILE_WIDTH = 64
    TILE_HEIGHT = 20
    TRACK_Y_AMP = 50
    TRACK_Y_FREQ = 0.05
    TRACK_GAP = 180
    
    MAX_EPISODE_SECONDS = 60
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # --- Persistent State (across resets) ---
        self.total_instruments_collected = 0
        self.unlocked_skins = [True, False, False, False]
        self.current_skin_index = 0
        self.skin_unlock_thresholds = [0, 5, 15, 30]

        # --- Game State (reset each episode) ---
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.gravity_dir = 1
        self.on_ground = False
        
        self.camera_x = 0.0
        self.steps = 0
        self.score = 0.0
        self.time_remaining = 0.0
        
        self.track_top = []
        self.track_bottom = []
        self.instruments = []
        self.finish_line_x = 0
        
        self.particles = []
        
        self.last_space_state = 0
        self.last_shift_state = 0
        
        self.tile_change_timer = 0
        self.tile_change_interval = 90 # 3 seconds at 30fps
        
        self.game_over_state = None # None, "win", "fall", "timeout"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Game State ---
        self.player_pos = pygame.math.Vector2(self.WIDTH / 4, self.HEIGHT / 2)
        self.player_vel = pygame.math.Vector2(self.PLAYER_BASE_SPEED_X, 0)
        self.gravity_dir = 1
        self.on_ground = False
        
        self.camera_x = 0.0
        self.steps = 0
        self.score = 0.0
        self.time_remaining = self.MAX_EPISODE_SECONDS
        
        self._generate_track()
        
        self.particles = []
        
        self.last_space_state = 0
        self.last_shift_state = 0
        
        self.tile_change_timer = 0
        self.tile_change_interval = 90
        
        self.game_over_state = None

        # Check for skin unlocks
        for i, threshold in enumerate(self.skin_unlock_thresholds):
            if self.total_instruments_collected >= threshold:
                self.unlocked_skins[i] = True

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False # Gymnasium standard
        
        if self.game_over_state:
            # If game is over, just return the final state
            return (
                self._get_observation(), 0, True, False, self._get_info()
            )

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Update Game Logic ---
        self.steps += 1
        self.time_remaining -= 1.0 / self.FPS
        
        # Action: Cycle skin
        if shift_held and not self.last_shift_state:
            # `sfx: skin_change_sound`
            self.current_skin_index = (self.current_skin_index + 1) % len(self.unlocked_skins)
            # Find next unlocked skin
            start_index = self.current_skin_index
            while not self.unlocked_skins[self.current_skin_index]:
                 self.current_skin_index = (self.current_skin_index + 1) % len(self.unlocked_skins)
                 if self.current_skin_index == start_index: # Prevent infinite loop if only one skin is unlocked
                    break
        
        # Action: Gravity Flip
        if space_held and not self.last_space_state and self.on_ground:
            # `sfx: gravity_flip_sound`
            self.gravity_dir *= -1
            self.player_vel.y = -self.PLAYER_FLIP_KICK * self.gravity_dir
            self.on_ground = False
            self._create_particles(self.player_pos, 20, (200, 200, 255), 2, 5)

        # Action: Steering
        if movement == 3: # Left
            self.player_vel.x -= self.PLAYER_STEER_FORCE
        elif movement == 4: # Right
            self.player_vel.x += self.PLAYER_STEER_FORCE
        
        # --- Physics Update ---
        self._update_player()
        
        # --- Collision & Interaction ---
        reward += self._handle_collisions_and_rewards()
        
        # --- Update Dynamic Elements ---
        self._update_tiles()
        self._update_particles()
        
        # --- Camera ---
        # Smooth follow camera
        target_camera_x = self.player_pos.x - self.WIDTH / 4
        self.camera_x = self.camera_x * 0.9 + target_camera_x * 0.1

        # --- Termination Check ---
        if self.player_pos.x >= self.finish_line_x:
            self.game_over_state = "win"
            reward += 100.0
            terminated = True
            # `sfx: win_jingle`
        elif self.player_pos.y > self.HEIGHT + 50 or self.player_pos.y < -50:
            self.game_over_state = "fall"
            reward -= 50.0
            terminated = True
            # `sfx: fall_sound`
        elif self.time_remaining <= 0:
            self.game_over_state = "timeout"
            reward -= 10.0
            terminated = True
            # `sfx: timeout_buzz`
            
        self.score += reward
        self.last_space_state = space_held
        self.last_shift_state = shift_held

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self):
        # Apply gravity if not on ground
        if not self.on_ground:
            self.player_vel.y += self.GRAVITY * self.gravity_dir
        
        # Apply drag and base speed
        self.player_vel.x *= self.PLAYER_DRAG
        if self.player_vel.x < self.PLAYER_BASE_SPEED_X:
            self.player_vel.x = self.PLAYER_BASE_SPEED_X
            
        # Clamp horizontal speed
        self.player_vel.x = max(0, min(self.player_vel.x, self.PLAYER_MAX_SPEED_X))
        
        # Update position
        self.player_pos += self.player_vel
        
        # Keep player within horizontal screen bounds (relative to camera)
        player_screen_x = self.player_pos.x - self.camera_x
        if player_screen_x < 0:
            self.player_pos.x = self.camera_x
            self.player_vel.x = max(self.player_vel.x, 0)
        if player_screen_x > self.WIDTH:
            self.player_pos.x = self.camera_x + self.WIDTH
            self.player_vel.x = min(self.player_vel.x, 0)

    def _handle_collisions_and_rewards(self):
        reward = 0.0
        self.on_ground = False
        
        # Forward movement reward
        reward += 0.01 * (self.player_vel.x / self.PLAYER_MAX_SPEED_X)

        # Determine which track to check against
        current_track = self.track_bottom if self.gravity_dir == 1 else self.track_top
        
        player_rect = self._get_player_rect()

        for tile in current_track:
            if tile['rect'].colliderect(player_rect):
                # Vertical collision
                if self.gravity_dir == 1 and self.player_vel.y > 0: # Moving down
                    self.player_pos.y = tile['rect'].top - player_rect.height / 2
                    self.player_vel.y = 0
                    self.on_ground = True
                elif self.gravity_dir == -1 and self.player_vel.y < 0: # Moving up
                    self.player_pos.y = tile['rect'].bottom + player_rect.height / 2
                    self.player_vel.y = 0
                    self.on_ground = True
                
                if self.on_ground:
                    # Tile interaction reward/penalty
                    if tile['type'] == 'green':
                        reward += 0.1
                        self.player_vel.x += 0.2
                        if self.np_random.random() < 0.2:
                             self._create_particles(self.player_pos, 3, self.TILE_COLORS['green'], 1, 2)
                    elif tile['type'] == 'blue':
                        reward -= 0.05
                        self.player_vel.x *= 0.98
                    # Red is neutral
                    break # Assume player is only on one tile at a time

        # Instrument collection
        collected_indices = []
        for i, instrument in enumerate(self.instruments):
            inst_pos, inst_rect = instrument
            if inst_rect.colliderect(player_rect):
                # `sfx: collect_instrument_sound`
                reward += 5.0
                self.total_instruments_collected += 1
                collected_indices.append(i)
                self._create_particles(inst_pos, 15, self.COLOR_INSTRUMENT, 3, 6)
        
        # Remove collected instruments
        for i in sorted(collected_indices, reverse=True):
            del self.instruments[i]
            
        return reward

    def _get_player_rect(self):
        # Using a simple rect for collision, visual can be different
        return pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 10, 20, 20)
        
    def _generate_track(self):
        self.track_top = []
        self.track_bottom = []
        self.instruments = []
        
        y_center = self.HEIGHT / 2
        
        for i in range(self.TRACK_LENGTH_TILES):
            x = i * self.TILE_WIDTH
            y_offset = self.TRACK_Y_AMP * math.sin(i * self.TRACK_Y_FREQ)
            
            # Bottom track
            bottom_y = y_center + self.TRACK_GAP / 2 + y_offset
            bottom_rect = pygame.Rect(x, bottom_y, self.TILE_WIDTH, self.TILE_HEIGHT)
            self.track_bottom.append({'rect': bottom_rect, 'type': random.choice(list(self.TILE_COLORS.keys()))})
            
            # Top track
            top_y = y_center - self.TRACK_GAP / 2 + y_offset - self.TILE_HEIGHT
            top_rect = pygame.Rect(x, top_y, self.TILE_WIDTH, self.TILE_HEIGHT)
            self.track_top.append({'rect': top_rect, 'type': random.choice(list(self.TILE_COLORS.keys()))})
            
            # Place instruments
            if i > 10 and i % 25 == 0 and self.np_random.random() < 0.7:
                on_top = self.np_random.choice([True, False])
                inst_y = (top_y - 20) if on_top else (bottom_y + self.TILE_HEIGHT + 20)
                inst_pos = pygame.math.Vector2(x + self.TILE_WIDTH/2, inst_y)
                inst_rect = pygame.Rect(inst_pos.x - 10, inst_pos.y - 10, 20, 20)
                self.instruments.append((inst_pos, inst_rect))

        self.finish_line_x = self.TRACK_LENGTH_TILES * self.TILE_WIDTH

    def _update_tiles(self):
        self.tile_change_timer += 1
        
        # Increase difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.tile_change_interval = max(20, self.tile_change_interval * 0.95)

        if self.tile_change_timer >= self.tile_change_interval:
            self.tile_change_timer = 0
            # `sfx: tile_change_chime`
            for tile in self.track_top:
                tile['type'] = random.choice(list(self.TILE_COLORS.keys()))
            for tile in self.track_bottom:
                tile['type'] = random.choice(list(self.TILE_COLORS.keys()))

    def _create_particles(self, pos, count, color, min_speed, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'lifespan': self.np_random.integers(15, 31), 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.9 # Drag
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Gradient background
        rect = pygame.Rect(0, 0, self.WIDTH, self.HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_BG_TOP, (0, 0, self.WIDTH, self.HEIGHT/2))
        pygame.draw.rect(self.screen, self.COLOR_BG_BOTTOM, (0, self.HEIGHT/2, self.WIDTH, self.HEIGHT/2))

    def _render_game_elements(self):
        # Pulsing effect for tiles
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 * 4 # 0 to 4
        
        # Render Tracks
        for track in [self.track_top, self.track_bottom]:
            for tile in track:
                screen_rect = tile['rect'].move(-self.camera_x, 0)
                if screen_rect.right < 0 or screen_rect.left > self.WIDTH:
                    continue
                
                color = self.TILE_COLORS[tile['type']]
                
                # Darker base for depth
                darker_color = tuple(max(0, c - 40) for c in color)
                pygame.draw.rect(self.screen, darker_color, screen_rect)
                
                # Pulsing inner rect
                inner_rect = screen_rect.inflate(-pulse, -pulse)
                pygame.draw.rect(self.screen, color, inner_rect, border_radius=2)

        # Render Instruments
        for pos, rect in self.instruments:
            screen_pos = (int(pos.x - self.camera_x), int(pos.y))
            if screen_pos[0] < -20 or screen_pos[0] > self.WIDTH + 20:
                continue
            
            # Simple musical note shape
            pygame.draw.circle(self.screen, self.COLOR_INSTRUMENT, (screen_pos[0] - 4, screen_pos[1] + 4), 6)
            pygame.draw.circle(self.screen, self.COLOR_INSTRUMENT, (screen_pos[0] + 4, screen_pos[1] + 2), 6)
            pygame.draw.line(self.screen, self.COLOR_INSTRUMENT, (screen_pos[0] + 2, screen_pos[1] + 4), (screen_pos[0] + 2, screen_pos[1] - 10), 3)
            pygame.draw.line(self.screen, self.COLOR_INSTRUMENT, (screen_pos[0] + 10, screen_pos[1] + 2), (screen_pos[0] + 10, screen_pos[1] - 8), 3)
            pygame.draw.line(self.screen, self.COLOR_INSTRUMENT, (screen_pos[0] + 2, screen_pos[1] - 10), (screen_pos[0] + 10, screen_pos[1] - 8), 3)


        # Render Finish Line
        finish_screen_x = self.finish_line_x - self.camera_x
        if 0 < finish_screen_x < self.WIDTH:
            for y in range(0, self.HEIGHT, 20):
                color = self.COLOR_FINISH_LINE if (y // 20) % 2 == 0 else (50, 50, 50)
                pygame.draw.rect(self.screen, color, (finish_screen_x, y, 10, 20))
                
        # Render Particles
        for p in self.particles:
            pos = (int(p['pos'].x - self.camera_x), int(p['pos'].y))
            size = max(1, int(p['lifespan'] / 6))
            pygame.draw.circle(self.screen, p['color'], pos, size)

        # Render Player
        self._render_player()

    def _render_player(self):
        screen_pos = pygame.math.Vector2(self.player_pos.x - self.camera_x, self.player_pos.y)
        
        # Glow effect
        glow_size = 25 + 5 * math.sin(self.steps * 0.3)
        temp_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, self.COLOR_PLAYER_GLOW, (glow_size, glow_size), glow_size)
        self.screen.blit(temp_surf, (int(screen_pos.x - glow_size), int(screen_pos.y - glow_size)))

        # Player vehicle skin
        skin_func = getattr(self, f"_draw_skin_{self.current_skin_index}", self._draw_skin_0)
        skin_func(screen_pos)

    def _draw_skin_0(self, pos): # Default Arrow
        points = [(-12, 8), (12, 0), (-12, -8)]
        angle = -math.degrees(math.atan2(self.player_vel.y, self.player_vel.x * self.gravity_dir))
        rotated_points = [pygame.math.Vector2(p).rotate(angle) for p in points]
        final_points = [(pos + p) for p in rotated_points]
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, final_points)
        pygame.draw.aalines(self.screen, self.COLOR_PLAYER, True, final_points)

    def _draw_skin_1(self, pos): # Saucer
        rect = pygame.Rect(pos.x - 12, pos.y - 5, 24, 10)
        pygame.draw.ellipse(self.screen, self.COLOR_PLAYER, rect)
        pygame.draw.ellipse(self.screen, (200, 200, 255), rect.inflate(-12, -2).move(0, -3))

    def _draw_skin_2(self, pos): # Orb
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, (int(pos.x), int(pos.y)), 10)
        inner_color = (255, 255, 255, 150)
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), 7, inner_color)

    def _draw_skin_3(self, pos): # Rocket
        body = [(-12, -6), (6, -6), (12, 0), (6, 6), (-12, 6)]
        angle = -math.degrees(math.atan2(self.player_vel.y, self.player_vel.x * self.gravity_dir))
        rotated_body = [pygame.math.Vector2(p).rotate(angle) for p in body]
        final_body = [(pos + p) for p in rotated_body]
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, final_body)
        # Flame
        flame_len = 5 + (self.player_vel.x - self.PLAYER_BASE_SPEED_X)
        flame_points = [(-12, -4), (-12 - flame_len, 0), (-12, 4)]
        rotated_flame = [pygame.math.Vector2(p).rotate(angle) for p in flame_points]
        final_flame = [(pos + p) for p in rotated_flame]
        pygame.draw.polygon(self.screen, (255, 100, 0), final_flame)


    def _render_ui(self):
        # Time remaining
        time_text = f"TIME: {max(0, self.time_remaining):.1f}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))
        
        # Speed
        speed_text = f"SPEED: {self.player_vel.x:.1f}"
        speed_surf = self.font_small.render(speed_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_surf, (10, self.HEIGHT - speed_surf.get_height() - 10))

        # Collected Instruments
        inst_text = f"INSTRUMENTS: {self.total_instruments_collected}"
        inst_surf = self.font_small.render(inst_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(inst_surf, (self.WIDTH - inst_surf.get_width() - 10, self.HEIGHT - inst_surf.get_height() - 10))
        
        # Score
        score_text = f"SCORE: {self.score:.0f}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.WIDTH // 2 - score_surf.get_width() // 2, 10))
        
        # Game Over Message
        if self.game_over_state:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            
            msg = ""
            if self.game_over_state == "win": msg = "FINISH!"
            elif self.game_over_state == "fall": msg = "FELL OFF TRACK"
            elif self.game_over_state == "timeout": msg = "TIME UP"
            
            msg_surf = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            overlay.blit(msg_surf, (self.WIDTH/2 - msg_surf.get_width()/2, self.HEIGHT/2 - msg_surf.get_height()/2 - 20))
            
            final_score_surf = self.font_small.render(f"Final Score: {self.score:.0f}", True, self.COLOR_UI_TEXT)
            overlay.blit(final_score_surf, (self.WIDTH/2 - final_score_surf.get_width()/2, self.HEIGHT/2 + 20))
            
            self.screen.blit(overlay, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "total_instruments": self.total_instruments_collected,
            "player_pos": (self.player_pos.x, self.player_pos.y),
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Pygame window for manual play
    # Re-enable display for manual testing
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Gravity Flip Rhythm Racer")
    clock = pygame.time.Clock()
    
    while running:
        # --- Action Mapping for Human ---
        action = [0, 0, 0] # no-op, space, shift
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0
                
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space, shift]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}")
            print("Press 'R' to restart.")
            # Wait for 'R' key press to reset
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_reset = False
                clock.tick(GameEnv.FPS)

        clock.tick(GameEnv.FPS)
        
    env.close()