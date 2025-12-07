
# Generated: 2025-08-27T21:55:59.540180
# Source Brief: brief_02954.md
# Brief Index: 2954

        
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

    user_guide = (
        "Controls: ←→ to move the cursor. Press space to hit the notes in the target zone."
    )

    game_description = (
        "A fast-paced rhythm game. Match the falling notes as they enter the target zone to complete the song and maximize your score."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.NUM_COLS = 4
        self.COL_WIDTH = self.WIDTH // self.NUM_COLS
        self.TOTAL_NOTES = 100
        self.MISS_LIMIT = 30
        self.MAX_STEPS = 30 * 120 # 2 minutes at 30fps

        # --- Visuals ---
        self.COLOR_BG = (15, 18, 26)
        self.COLOR_GRID = (40, 45, 60)
        self.COLOR_GRID_BEAT = (70, 80, 100)
        self.COLOR_TARGET_ZONE = (255, 215, 0, 50) # Yellow, semi-transparent
        self.COLOR_CURSOR = (0, 191, 255) # Deep Sky Blue
        self.COLOR_SUCCESS = (0, 255, 127) # Spring Green
        self.COLOR_MISS = (255, 69, 0) # OrangeRed
        self.NOTE_COLORS = [
            (255, 0, 128),   # Pink
            (0, 255, 255),   # Cyan
            (128, 0, 255),   # Purple
            (255, 128, 0),   # Orange
        ]

        self.HIT_ZONE_Y = 320
        self.HIT_ZONE_HEIGHT = 50
        self.NOTE_HEIGHT = 15

        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_col = 0
        self.last_space_held = False
        self.song_data = []
        self.song_cursor = 0
        self.active_notes = []
        self.particles = []
        self.combo = 0
        self.missed_notes = 0
        self.hit_notes_count = 0
        self.note_speed = 0.0
        self.last_difficulty_milestone = 0
        self.beat_flash_timer = 0
        self.feedback_messages = []

        self.reset()
        self.validate_implementation()

    def _generate_song(self):
        song = []
        current_frame = 60 # Start with a small delay
        for i in range(self.TOTAL_NOTES):
            # Ensure notes don't spawn too close together
            current_frame += self.np_random.integers(20, 45)
            col = self.np_random.integers(0, self.NUM_COLS)
            song.append({"spawn_frame": current_frame, "col": col, "id": i})
        return song

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_col = self.NUM_COLS // 2
        self.last_space_held = False

        self.song_data = self._generate_song()
        self.song_cursor = 0
        self.active_notes = []
        self.particles = []
        
        self.combo = 0
        self.missed_notes = 0
        self.hit_notes_count = 0
        self.note_speed = 2.0
        self.last_difficulty_milestone = 0
        self.beat_flash_timer = 0
        self.feedback_messages = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        self.steps += 1
        self.beat_flash_timer = max(0, self.beat_flash_timer - 1)

        # --- 1. Unpack Action and Handle Input ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held
        
        if movement == 3: # Left
            self.cursor_col = max(0, self.cursor_col - 1)
        elif movement == 4: # Right
            self.cursor_col = min(self.NUM_COLS - 1, self.cursor_col + 1)
        
        # --- 2. Update Game State ---
        self._update_song_spawning()
        self._update_particles()
        
        # --- 3. Process Notes and Player Actions for Rewards ---
        
        # Penalty for cursor not being in a column with an active note
        active_cols = {note['col'] for note in self.active_notes}
        if active_cols and self.cursor_col not in active_cols:
            reward -= 0.01

        # Handle player's match attempt
        if space_pressed:
            hit_this_frame = self._handle_hit_attempt()
            if hit_this_frame:
                reward += hit_this_frame['reward']
                self.score += hit_this_frame['score']
                self.combo = hit_this_frame['combo']
                self.hit_notes_count += 1
                # SFX: Play successful hit sound
            else:
                # Penalty for pressing at the wrong time/place
                reward -= 0.2
                self.combo = 0
                self._add_feedback("Miss!", self.cursor_col, self.COLOR_MISS)
                # SFX: Play miss/error sound

        # Update and check for missed notes
        missed_this_frame = self._update_notes_position()
        if missed_this_frame:
            self.missed_notes += len(missed_this_frame)
            reward -= 1.0 * len(missed_this_frame)
            self.combo = 0
            # SFX: Play miss sound for each note

        # --- 4. Update Difficulty ---
        current_milestone = self.hit_notes_count // 20
        if current_milestone > self.last_difficulty_milestone:
            self.note_speed += 0.25
            self.last_difficulty_milestone = current_milestone
            # SFX: Play level-up sound

        # --- 5. Check Termination Conditions ---
        terminated = False
        if self.missed_notes >= self.MISS_LIMIT:
            terminated = True
            reward -= 50
            self.game_over = True
        elif self.song_cursor >= len(self.song_data) and not self.active_notes:
            terminated = True
            reward += 50
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_song_spawning(self):
        if self.song_cursor < len(self.song_data) and self.steps >= self.song_data[self.song_cursor]['spawn_frame']:
            note_info = self.song_data[self.song_cursor]
            self.active_notes.append({
                "y": 0,
                "col": note_info['col'],
                "id": note_info['id']
            })
            self.song_cursor += 1
            # SFX: Play note spawn sound (subtle)

    def _update_notes_position(self):
        missed_notes = []
        notes_to_keep = []
        for note in self.active_notes:
            note['y'] += self.note_speed
            if note['y'] > self.HEIGHT:
                missed_notes.append(note)
                self._add_feedback("Miss!", note['col'], self.COLOR_MISS)
            else:
                notes_to_keep.append(note)
        self.active_notes = notes_to_keep
        return missed_notes

    def _handle_hit_attempt(self):
        for i, note in enumerate(self.active_notes):
            if note['col'] == self.cursor_col:
                hit_zone_top = self.HIT_ZONE_Y - self.NOTE_HEIGHT / 2
                hit_zone_bottom = self.HIT_ZONE_Y + self.HIT_ZONE_HEIGHT - self.NOTE_HEIGHT / 2
                if hit_zone_top <= note['y'] <= hit_zone_bottom:
                    # Successful Hit
                    dist_from_center = abs(note['y'] - (self.HIT_ZONE_Y + self.HIT_ZONE_HEIGHT/2 - self.NOTE_HEIGHT/2))
                    
                    # Determine hit quality
                    if dist_from_center < self.HIT_ZONE_HEIGHT * 0.2:
                        quality_text, quality_score, quality_reward = "Perfect!", 100, 2.0
                    elif dist_from_center < self.HIT_ZONE_HEIGHT * 0.4:
                        quality_text, quality_score, quality_reward = "Great!", 50, 1.0
                    else:
                        quality_text, quality_score, quality_reward = "Good", 25, 0.5
                    
                    self._add_feedback(quality_text, note['col'], self.COLOR_SUCCESS)

                    base_reward = quality_reward
                    combo_reward = 1.0 if self.combo > 0 else 5.0 # Combo start bonus
                    
                    new_combo = self.combo + 1
                    score_gain = quality_score * new_combo
                    
                    self._spawn_particles(self.COL_WIDTH * (note['col'] + 0.5), note['y'] + self.NOTE_HEIGHT/2, self.NOTE_COLORS[note['col']])
                    self.beat_flash_timer = 5 # Flash grid on successful hit
                    
                    # Remove note
                    del self.active_notes[i]
                    
                    return {
                        'reward': base_reward + combo_reward,
                        'score': score_gain,
                        'combo': new_combo
                    }
        return None # No note was hit

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['dx']
            p['y'] += p['dy']
            p['dy'] += 0.1 # Gravity
            p['life'] -= 1

    def _spawn_particles(self, x, y, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': x, 'y': y,
                'dx': math.cos(angle) * speed, 'dy': math.sin(angle) * speed,
                'life': self.np_random.integers(15, 30),
                'color': color
            })
            
    def _add_feedback(self, text, col, color):
        x = self.COL_WIDTH * (col + 0.5)
        y = self.HIT_ZONE_Y - 20
        self.feedback_messages.append({'text': text, 'x': x, 'y': y, 'color': color, 'life': 30})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render target zone
        target_surface = pygame.Surface((self.WIDTH, self.HIT_ZONE_HEIGHT), pygame.SRCALPHA)
        target_surface.fill(self.COLOR_TARGET_ZONE)
        self.screen.blit(target_surface, (0, self.HIT_ZONE_Y))

        # Render grid lines
        grid_color = self.COLOR_GRID_BEAT if self.beat_flash_timer > 0 else self.COLOR_GRID
        for i in range(1, self.NUM_COLS):
            pygame.draw.line(self.screen, grid_color, (i * self.COL_WIDTH, 0), (i * self.COL_WIDTH, self.HEIGHT), 1)

        # Render cursor
        cursor_rect = pygame.Rect(self.cursor_col * self.COL_WIDTH, self.HIT_ZONE_Y, self.COL_WIDTH, self.HIT_ZONE_HEIGHT)
        pygame.gfxdraw.box(self.screen, cursor_rect, (*self.COLOR_CURSOR, 80))
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=5)

        # Render notes
        for note in self.active_notes:
            note_rect = pygame.Rect(
                note['col'] * self.COL_WIDTH + 5,
                int(note['y']),
                self.COL_WIDTH - 10,
                self.NOTE_HEIGHT
            )
            pygame.draw.rect(self.screen, self.NOTE_COLORS[note['col']], note_rect, border_radius=5)
            pygame.draw.rect(self.screen, (255, 255, 255), note_rect, 2, border_radius=5)

        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), 2, color)
            
        # Render feedback messages
        new_feedback = []
        for msg in self.feedback_messages:
            msg['life'] -= 1
            if msg['life'] > 0:
                alpha = max(0, min(255, int(255 * (msg['life'] / 30.0))))
                text_surf = self.font_main.render(msg['text'], True, msg['color'])
                text_surf.set_alpha(alpha)
                text_rect = text_surf.get_rect(center=(int(msg['x']), int(msg['y'])))
                self.screen.blit(text_surf, text_rect)
                msg['y'] -= 1 # Move text up
                new_feedback.append(msg)
        self.feedback_messages = new_feedback

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        # Combo
        if self.combo > 1:
            combo_text = self.font_main.render(f"COMBO: {self.combo}x", True, self.COLOR_CURSOR)
            text_rect = combo_text.get_rect(topright=(self.WIDTH - 10, 10))
            self.screen.blit(combo_text, text_rect)

        # Notes remaining / Misses
        notes_text = self.font_small.render(
            f"NOTES: {self.hit_notes_count}/{self.TOTAL_NOTES} | MISSES: {self.missed_notes}/{self.MISS_LIMIT}", True, (200, 200, 200)
        )
        text_rect = notes_text.get_rect(midbottom=(self.WIDTH / 2, self.HEIGHT - 10))
        self.screen.blit(notes_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "combo": self.combo,
            "missed_notes": self.missed_notes,
            "hit_notes": self.hit_notes_count,
        }

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # Use a dummy screen for display if running this file
    pygame.display.set_caption("Rhythm Game")
    display_screen = pygame.display.set_mode((640, 400))
    
    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        # --- Human Input to Action Conversion ---
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
            
        action = np.array([movement, space_held, shift_held])
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset()
                terminated = False

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            # On game over, wait for reset key
            pass

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Run at 30 FPS

    env.close()