
# Generated: 2025-08-27T18:32:37.410912
# Source Brief: brief_01862.md
# Brief Index: 1862

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to hit the notes in the corresponding lane as they cross the white line."
    )

    game_description = (
        "A retro-futuristic rhythm racer. Hit the notes on beat to build combos and speed towards the finish line. Miss too many and you fail!"
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.HIT_ZONE_X = 120
        self.HIT_TOLERANCE = 18
        self.MAX_MISSES = 3
        self.TOTAL_NOTES = 50
        self.FINISH_LINE_DISTANCE = 10000
        self.MAX_STEPS = 2000
        self.NOTE_SPACING_MIN = 150
        self.NOTE_SPACING_MAX = 250
        
        # Colors (Neon/Retro-Futuristic)
        self.COLOR_BG = (15, 10, 40)
        self.COLOR_LANE = (40, 30, 80)
        self.COLOR_LANE_HI = (60, 50, 120)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_NOTE = (0, 200, 255)
        self.COLOR_NOTE_GLOW = (0, 200, 255, 50)
        self.COLOR_HIT = (50, 255, 50)
        self.COLOR_MISS = (255, 50, 50)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_UI_BG = (25, 20, 50, 180)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_ui = pygame.font.Font(None, 28)
        self.font_feedback = pygame.font.Font(None, 24)
        self.font_combo = pygame.font.Font(None, 40)
        
        # Lane and action mapping
        self.lane_ys = [
            self.SCREEN_HEIGHT // 2 - 120,
            self.SCREEN_HEIGHT // 2 - 40,
            self.SCREEN_HEIGHT // 2 + 40,
            self.SCREEN_HEIGHT // 2 + 120,
        ]
        self.action_to_lane = {1: 0, 3: 1, 4: 2, 2: 3} # Up, Left, Right, Down

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.progress = 0.0
        self.note_speed = 0.0
        self.missed_notes = 0
        self.combo = 0
        self.successful_hits = 0
        self.notes = []
        self.particles = []
        self.hit_feedback = []
        self.stars = []
        self.last_action_time = -100

        self.reset()
        self.validate_implementation()
    
    def _generate_notes(self):
        notes = []
        current_x = 800  # Start first note off-screen
        for i in range(self.TOTAL_NOTES):
            lane = self.np_random.integers(0, 4)
            spacing = self.np_random.uniform(self.NOTE_SPACING_MIN, self.NOTE_SPACING_MAX)
            current_x += spacing
            notes.append({'x': current_x, 'lane': lane, 'id': i, 'hit': False})
        return notes
        
    def _generate_stars(self):
        self.stars = []
        for _ in range(100):
            self.stars.append({
                'x': self.np_random.uniform(0, self.SCREEN_WIDTH),
                'y': self.np_random.uniform(0, self.SCREEN_HEIGHT),
                'speed': self.np_random.uniform(0.1, 0.5)
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.progress = 0.0
        self.note_speed = 4.0
        self.missed_notes = 0
        self.combo = 0
        self.successful_hits = 0
        
        self.notes = self._generate_notes()
        self._generate_stars()
        
        self.particles = []
        self.hit_feedback = []
        self.last_action_time = -100
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # Unpack factorized action
        movement = action[0]
        
        # --- Game Logic ---
        
        # Update world progress
        self.progress += self.note_speed

        # Update entities
        self._update_notes()
        self._update_particles()
        self._update_feedback()
        self._update_stars()
        
        # Check for misses
        for note in self.notes:
            if note['x'] < self.HIT_ZONE_X - self.HIT_TOLERANCE and not note['hit']:
                note['hit'] = True # Mark as handled
                self.missed_notes += 1
                self.combo = 0
                reward -= 1
                self._create_feedback("MISS", self.action_to_lane.get(movement, note['lane']), self.COLOR_MISS)
                # sfx: miss_sound
        
        # Handle player input
        if movement > 0:
            self.last_action_time = self.steps
            hit_lane = self.action_to_lane.get(movement)
            if hit_lane is not None:
                hit_found = False
                for note in self.notes:
                    if not note['hit'] and note['lane'] == hit_lane and abs(note['x'] - self.HIT_ZONE_X) <= self.HIT_TOLERANCE:
                        note['hit'] = True
                        hit_found = True
                        
                        self.score += 10 * (1 + self.combo // 5)
                        self.combo += 1
                        self.successful_hits += 1
                        reward += 1
                        
                        if self.combo > 0 and self.combo % 5 == 0:
                            reward += 5
                        
                        # Difficulty scaling
                        if self.successful_hits > 0 and self.successful_hits % 10 == 0:
                            self.note_speed += 0.5
                        
                        self._create_particles(hit_lane, self.COLOR_HIT)
                        self._create_feedback("PERFECT", hit_lane, self.COLOR_HIT)
                        # sfx: hit_sound
                        break
                
                # if not hit_found:
                #     # Optional: penalty for hitting nothing
                #     self._create_particles(hit_lane, self.COLOR_MISS, count=5)
                #     # sfx: whiff_sound

        # Clean up old notes that are off-screen
        self.notes = [n for n in self.notes if n['x'] > -20]

        # --- Termination ---
        terminated = False
        if self.missed_notes > self.MAX_MISSES:
            terminated = True
            reward -= 100
            self.game_over = True
        
        # Victory if all notes are cleared and at least one was hit
        if not self.notes and self.successful_hits > 0:
            terminated = True
            reward += 100
            self.game_over = True
            
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_notes(self):
        for note in self.notes:
            note['x'] -= self.note_speed

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _update_feedback(self):
        for f in self.hit_feedback:
            f['pos'][1] -= 0.5
            f['life'] -= 1
        self.hit_feedback = [f for f in self.hit_feedback if f['life'] > 0]
        
    def _update_stars(self):
        for star in self.stars:
            star['x'] -= star['speed'] * self.note_speed * 0.2
            if star['x'] < 0:
                star['x'] = self.SCREEN_WIDTH
                star['y'] = self.np_random.uniform(0, self.SCREEN_HEIGHT)

    def _create_particles(self, lane_idx, color, count=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': [self.HIT_ZONE_X, self.lane_ys[lane_idx]],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'color': color,
                'size': self.np_random.uniform(1, 4)
            })

    def _create_feedback(self, text, lane_idx, color):
        self.hit_feedback.append({
            'text': text,
            'pos': [self.HIT_ZONE_X + 20, self.lane_ys[lane_idx]],
            'life': 30,
            'color': color
        })
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_stars()
        self._render_lanes()
        self._render_notes()
        self._render_player_zone()
        self._render_particles()
        self._render_feedback_text()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_stars(self):
        for star in self.stars:
            size = int(star['speed'] * 2)
            color_val = int(star['speed'] * 150) + 50
            pygame.draw.rect(self.screen, (color_val, color_val, color_val), (int(star['x']), int(star['y']), size, size))

    def _render_lanes(self):
        action_time_fade = max(0, 1.0 - (self.steps - self.last_action_time) / 10.0)
        
        for i, y in enumerate(self.lane_ys):
            is_active_lane = False
            if action_time_fade > 0:
                movement = self.action_space.sample()[0] # A bit of a hack to get recent action
                if self.action_to_lane.get(movement) == i:
                    is_active_lane = True

            color = self.COLOR_LANE_HI if is_active_lane else self.COLOR_LANE
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y), 2)

    def _render_notes(self):
        for note in self.notes:
            if not note['hit']:
                x, y = int(note['x']), self.lane_ys[note['lane']]
                size = 12
                
                # Glow effect
                glow_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, self.COLOR_NOTE_GLOW, (size, size), size)
                self.screen.blit(glow_surf, (x - size, y - size), special_flags=pygame.BLEND_RGBA_ADD)

                # Core note
                pygame.gfxdraw.filled_circle(self.screen, x, y, size - 2, self.COLOR_NOTE)
                pygame.gfxdraw.aacircle(self.screen, x, y, size - 2, self.COLOR_NOTE)

    def _render_player_zone(self):
        # Pulsing hit line
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 * 100 + 155
        color = (int(pulse), int(pulse), 255)
        pygame.draw.line(self.screen, color, (self.HIT_ZONE_X, 0), (self.HIT_ZONE_X, self.SCREEN_HEIGHT), 3)
        
        # Player marker triangle
        p1 = (self.HIT_ZONE_X - 15, self.SCREEN_HEIGHT // 2 - 10)
        p2 = (self.HIT_ZONE_X - 15, self.SCREEN_HEIGHT // 2 + 10)
        p3 = (self.HIT_ZONE_X - 25, self.SCREEN_HEIGHT // 2)
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [p1, p2, p3])

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            alpha = int(255 * (p['life'] / 30.0))
            color = (*p['color'], alpha)
            
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0,0,p['size'], p['size']))
            self.screen.blit(temp_surf, (pos[0] - p['size']//2, pos[1] - p['size']//2), special_flags=pygame.BLEND_RGBA_ADD)


    def _render_feedback_text(self):
        for f in self.hit_feedback:
            alpha = max(0, min(255, int(255 * (f['life'] / 20.0))))
            text_surf = self.font_feedback.render(f['text'], True, f['color'])
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, (int(f['pos'][0]), int(f['pos'][1])))

    def _render_ui(self):
        # UI Background panels
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (5, 5, 180, 35), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (self.SCREEN_WIDTH - 155, 5, 150, 35), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (self.SCREEN_WIDTH // 2 - 100, self.SCREEN_HEIGHT - 40, 200, 35), border_radius=5)

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 12))
        
        # Combo
        if self.combo > 1:
            combo_text = self.font_combo.render(f"{self.combo}x", True, self.COLOR_HIT)
            text_rect = combo_text.get_rect(center=(self.SCREEN_WIDTH / 2, 40))
            self.screen.blit(combo_text, text_rect)

        # Notes remaining
        notes_left = len([n for n in self.notes if not n['hit']])
        notes_text = self.font_ui.render(f"NOTES: {notes_left}/{self.TOTAL_NOTES}", True, self.COLOR_TEXT)
        self.screen.blit(notes_text, (self.SCREEN_WIDTH - 145, 12))

        # Misses
        miss_text = self.font_ui.render(f"MISSES: {self.missed_notes} / {self.MAX_MISSES + 1}", True, self.COLOR_TEXT)
        miss_rect = miss_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT - 23))
        self.screen.blit(miss_text, miss_rect)

        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            end_text_str = "FINISH!" if self.missed_notes <= self.MAX_MISSES else "FAILED"
            end_text_color = self.COLOR_HIT if self.missed_notes <= self.MAX_MISSES else self.COLOR_MISS
            
            end_text = pygame.font.Font(None, 80).render(end_text_str, True, end_text_color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "combo": self.combo,
            "missed_notes": self.missed_notes,
            "note_speed": self.note_speed,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Rhythm Racer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    action = env.action_space.sample()
    action.fill(0)

    while running:
        # Human input
        movement_action = 0 # No-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        elif keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
        
        action = np.array([movement_action, 0, 0]) # Space and Shift not used

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()