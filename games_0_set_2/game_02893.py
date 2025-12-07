
# Generated: 2025-08-28T06:21:45.485402
# Source Brief: brief_02893.md
# Brief Index: 2893

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move your cursor. Press space to hit the notes on the beat."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, neon-drenched rhythm game. Hit the falling notes on the beat to rack up points and climb the combo meter."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Screen dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.game_font = pygame.font.Font(None, 36)
        self.combo_font = pygame.font.Font(None, 48)
        self.title_font = pygame.font.Font(None, 60)

        # Colors
        self.COLOR_BG_TOP = (10, 5, 25)
        self.COLOR_BG_BOTTOM = (20, 10, 40)
        self.COLOR_GRID = (40, 100, 120)
        self.COLOR_CURSOR = (255, 255, 220)
        self.COLOR_TEXT = (220, 220, 255)
        self.NOTE_COLORS = [
            (255, 50, 150),  # Neon Pink
            (50, 255, 255),  # Neon Cyan
            (50, 255, 100),  # Neon Green
            (255, 150, 50),  # Neon Orange
        ]

        # Game constants
        self.NUM_LANES = 4
        self.MAX_BEATS = 180
        self.MAX_MISSES = 36
        self.HIT_ZONE_Y = 350
        self.NOTE_SPEED_SCALE = 150  # Visual speed multiplier
        self.HIT_TOLERANCE = 25 # Pixels
        self.MOVE_COOLDOWN = 0.1 # Seconds
        
        # Initialize state variables
        self.state_vars = [
            'steps', 'score', 'game_over', 'beat_count', 'hits', 'misses', 'combo',
            'tempo', 'beat_interval', 'time_since_beat', 'cursor_pos', 'notes',
            'particles', 'song_pattern', 'space_was_held', 'move_cooldown_timer',
            'last_combo_pop'
        ]
        for var in self.state_vars:
            setattr(self, var, None)
        
        self.reset()
        self._validate_implementation()
    
    def _generate_song(self):
        """Generates a pseudo-random song pattern for consistency."""
        self.song_pattern = [[] for _ in range(self.MAX_BEATS + 20)]
        num_notes = 0
        for i in range(5, self.MAX_BEATS):
            # Increase note density over time
            p_single = 0.20 + 0.30 * (i / self.MAX_BEATS)
            p_double = 0.01 + 0.10 * (i / self.MAX_BEATS)
            
            rand_val = self.np_random.random()
            
            if rand_val < p_double and i > 20: # Doubles only after beat 20
                lanes = self.np_random.choice(self.NUM_LANES, 2, replace=False).tolist()
                self.song_pattern[i] = lanes
                num_notes += 2
            elif rand_val < p_double + p_single:
                lane = self.np_random.integers(0, self.NUM_LANES)
                self.song_pattern[i] = [lane]
                num_notes += 1
        
        # Ensure total notes is roughly 180 for accuracy calculations
        # This is a rough approximation, actual misses depend on play
        self.total_notes_in_song = num_notes


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.beat_count = 0
        self.hits = 0
        self.misses = 0
        self.combo = 0
        self.last_combo_pop = 0
        
        self.tempo = 1.0  # beats per second
        self.beat_interval = 1.0 / self.tempo
        self.time_since_beat = 0.0
        
        self.cursor_pos = self.NUM_LANES // 2
        self.notes = []
        self.particles = []

        self.space_was_held = True # Prevent tap on first frame
        self.move_cooldown_timer = 0.0
        
        self._generate_song()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        dt = self.clock.tick(30) / 1000.0
        self.steps += 1
        self.move_cooldown_timer = max(0, self.move_cooldown_timer - dt)

        # 1. Unpack and handle player input
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        if self.move_cooldown_timer == 0:
            moved = False
            if movement == 3: # Left
                self.cursor_pos = max(0, self.cursor_pos - 1)
                moved = True
            elif movement == 4: # Right
                self.cursor_pos = min(self.NUM_LANES - 1, self.cursor_pos + 1)
                moved = True
            if moved:
                self.move_cooldown_timer = self.MOVE_COOLDOWN
        
        is_tap = space_held and not self.space_was_held
        self.space_was_held = space_held

        # 2. Update beat and timing
        self.time_since_beat += dt
        beat_processed_this_frame = False
        while self.time_since_beat >= self.beat_interval and not self.game_over:
            beat_processed_this_frame = True
            self.time_since_beat -= self.beat_interval
            self.beat_count += 1

            # Check for missed notes (notes that passed the hit zone on the previous beat)
            surviving_notes = []
            for note in self.notes:
                if note['y'] > self.HIT_ZONE_Y + self.HIT_TOLERANCE:
                    self.misses += 1
                    self.combo = 0
                    reward -= 1
                    # sound: miss_sound.play()
                else:
                    surviving_notes.append(note)
            self.notes = surviving_notes

            # Spawn new notes for the current beat
            if self.beat_count < len(self.song_pattern):
                for lane in self.song_pattern[self.beat_count]:
                    self.notes.append({'lane': lane, 'y': 0, 'beat': self.beat_count})

            # Update tempo
            if self.beat_count > 0 and self.beat_count % 30 == 0:
                self.tempo = min(4.0, self.tempo + 0.05)
                self.beat_interval = 1.0 / self.tempo
        
        # 3. Handle Hit Logic
        if is_tap:
            hit_found = False
            for note in sorted(self.notes, key=lambda n: abs(n['y'] - self.HIT_ZONE_Y)):
                if note['lane'] == self.cursor_pos:
                    if abs(note['y'] - self.HIT_ZONE_Y) <= self.HIT_TOLERANCE:
                        hit_found = True
                        self.notes.remove(note)
                        self.hits += 1
                        self.combo += 1
                        self.score += 10 * self.combo
                        reward += 1
                        # sound: hit_sound.play()
                        if self.combo > 0 and self.combo % 10 == 0:
                            reward += 5
                            self.last_combo_pop = 1.0 # For animation
                        
                        self._create_particles(self.cursor_pos)
                        break # Only hit one note per tap
            # if not hit_found:
                # No penalty for tapping on empty space

        # 4. Update object positions
        for note in self.notes:
            note['y'] += self.tempo * self.NOTE_SPEED_SCALE * dt
        
        for p in self.particles[:]:
            p['x'] += p['vx'] * dt
            p['y'] += p['vy'] * dt
            p['life'] -= dt
            if p['life'] <= 0:
                self.particles.remove(p)
        
        self.last_combo_pop = max(0, self.last_combo_pop - dt * 2)

        # 5. Check for termination
        terminated = False
        if self.misses > self.MAX_MISSES:
            reward -= 50
            terminated = True
        if self.beat_count >= self.MAX_BEATS:
            terminated = True
            # Final accuracy calculation
            total_judged = self.hits + self.misses
            accuracy = self.hits / total_judged if total_judged > 0 else 1.0
            if accuracy >= 0.8:
                reward += 50
            else:
                reward -= 50
        
        if terminated:
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particles(self, lane):
        lane_width = (self.SCREEN_WIDTH * 0.6) / self.NUM_LANES
        x = (self.SCREEN_WIDTH * 0.2) + (lane + 0.5) * lane_width
        y = self.HIT_ZONE_Y
        color = self.NOTE_COLORS[lane]
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(50, 150)
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': random.uniform(0.3, 0.7),
                'color': color,
                'radius': random.uniform(2, 5)
            })

    def _render_game(self):
        # Draw gradient background
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        # Define highway geometry
        highway_width = self.SCREEN_WIDTH * 0.6
        lane_width = highway_width / self.NUM_LANES
        highway_left = (self.SCREEN_WIDTH - highway_width) / 2

        # Draw hit zone grid and pulse effect
        pulse = abs(math.sin(self.beat_count * math.pi + (self.time_since_beat/self.beat_interval)*math.pi))
        hit_zone_color = (
            min(255, self.COLOR_GRID[0] + pulse * 40),
            min(255, self.COLOR_GRID[1] + pulse * 40),
            min(255, self.COLOR_GRID[2] + pulse * 40),
        )
        
        hit_zone_rect = pygame.Rect(highway_left, self.HIT_ZONE_Y - 30, highway_width, 60)
        pygame.draw.rect(self.screen, (0,0,0,100), hit_zone_rect)
        pygame.draw.rect(self.screen, hit_zone_color, hit_zone_rect, 2)


        # Draw lanes
        for i in range(self.NUM_LANES + 1):
            x = highway_left + i * lane_width
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        
        # Draw cursor
        cursor_x = highway_left + self.cursor_pos * lane_width
        cursor_rect = pygame.Rect(cursor_x, self.HIT_ZONE_Y - 30, lane_width, 60)
        
        # Glow effect for cursor
        glow_surface = pygame.Surface((lane_width + 20, 80), pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*self.COLOR_CURSOR, 50), glow_surface.get_rect(), border_radius=10)
        self.screen.blit(glow_surface, (cursor_x - 10, self.HIT_ZONE_Y - 40), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=5)

        # Draw notes
        for note in self.notes:
            note_x = highway_left + note['lane'] * lane_width
            note_y = note['y']
            color = self.NOTE_COLORS[note['lane']]
            
            note_rect = pygame.Rect(note_x, note_y - 10, lane_width, 20)
            
            # Glow effect for notes
            glow_color = (*color, 100)
            pygame.gfxdraw.box(self.screen, note_rect.inflate(8, 8), (*glow_color, 80))
            
            pygame.draw.rect(self.screen, color, note_rect)
            pygame.draw.rect(self.screen, (255,255,255), note_rect, 1)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 0.7))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (p['x']-p['radius'], p['y']-p['radius']), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score
        score_text = self.game_font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Accuracy
        total_judged = self.hits + self.misses
        accuracy = (self.hits / total_judged * 100) if total_judged > 0 else 100.0
        acc_text = self.game_font.render(f"Acc: {accuracy:.1f}%", True, self.COLOR_TEXT)
        self.screen.blit(acc_text, (self.SCREEN_WIDTH - acc_text.get_width() - 10, 10))
        
        # Song Progress
        progress = self.beat_count / self.MAX_BEATS
        bar_width = self.SCREEN_WIDTH - 20
        pygame.draw.rect(self.screen, self.COLOR_GRID, (10, self.SCREEN_HEIGHT - 20, bar_width, 10), 1)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (10, self.SCREEN_HEIGHT - 20, bar_width * progress, 10))


        # Combo
        if self.combo > 1:
            pop_scale = 1.0 + 0.5 * self.last_combo_pop
            font_size = int(48 * pop_scale)
            
            current_font = pygame.font.Font(None, font_size)
            combo_str = f"{self.combo} COMBO"
            combo_text = current_font.render(combo_str, True, self.COLOR_TEXT)
            
            text_rect = combo_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 60))
            self.screen.blit(combo_text, text_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            total_judged = self.hits + self.misses
            accuracy = self.hits / total_judged if total_judged > 0 else 1.0
            
            result_text_str = "SONG CLEARED" if accuracy >= 0.8 and self.beat_count >= self.MAX_BEATS else "SONG FAILED"
            result_text = self.title_font.render(result_text_str, True, self.COLOR_CURSOR)
            result_rect = result_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
            self.screen.blit(result_text, result_rect)

            final_score_text = self.game_font.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 30))
            self.screen.blit(final_score_text, final_score_rect)


    def _get_observation(self):
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.beat_count,
            "hits": self.hits,
            "misses": self.misses,
            "combo": self.combo,
            "tempo": self.tempo,
        }

    def close(self):
        pygame.quit()

    def _validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")