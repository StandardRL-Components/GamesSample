import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:49:06.616528
# Source Brief: brief_01301.md
# Brief Index: 1301
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A rhythm-based game where you guide a falling geometric shape. "
        "Match the shape's position and size to the target on the beat line to score points."
    )
    user_guide = (
        "Use ←→ arrow keys to move the shape left/right and ↑↓ to change its size. "
        "Match the target on the line to score."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_LINE_Y = 350
    MAX_STEPS = 2000
    MAX_MISSES = 10

    # Colors
    COLOR_BG_TOP = (15, 20, 40)
    COLOR_BG_BOTTOM = (30, 40, 60)
    COLOR_WORD = (255, 255, 0)
    COLOR_WORD_GLOW = (255, 255, 0, 50)
    COLOR_TARGET = (0, 180, 255)
    COLOR_TARGET_GLOW = (0, 180, 255, 40)
    COLOR_OBSTACLE = (100, 110, 130)
    COLOR_PARTICLE_HIT = (50, 255, 150)
    COLOR_PARTICLE_MISS = (255, 50, 100)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_TIMELINE = (60, 70, 90)
    COLOR_BEAT_MARKER = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_word = pygame.font.SysFont("Arial", 18, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.missed_hits = 0
        self.game_over = False
        
        self.tempo = 0.0
        self.obstacle_frequency = 0
        self.fall_speed = 0.0
        
        self.active_word = None
        self.target_note = None
        self.obstacles = []
        self.particles = []
        
        self.note_track = []
        self.track_idx = 0
        self.beat_progress = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.missed_hits = 0
        self.game_over = False

        # --- Difficulty Parameters ---
        self.tempo = 1.0  # beats per step
        self.obstacle_frequency = 1
        self.fall_speed = 2.5
        self.next_obstacle_increase = 500
        self.next_tempo_increase = 1000

        # --- Entity Lists ---
        self.active_word = None
        self.target_note = None
        self.obstacles = []
        self.particles = []

        # --- Rhythm Track ---
        self._generate_track()
        self.track_idx = 0
        
        # --- Spawn initial elements ---
        self._spawn_next_pair()
        self._spawn_obstacles()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, _, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.steps += 1

        # --- Update Game Logic ---
        self._update_difficulty()
        self._update_word(movement)
        self._update_particles()
        
        # --- Collision and Event Logic ---
        if self.active_word:
            # Proximity reward
            reward += self._calculate_proximity_reward()

            # Obstacle collision
            for obs in self.obstacles:
                if self.active_word['rect'].colliderect(obs):
                    # SFX: Obstacle hit sound
                    reward -= 5
                    self.missed_hits += 1
                    self._create_particles(self.active_word['pos'], self.COLOR_PARTICLE_MISS, 20)
                    self._spawn_next_pair()
                    break
            
            # Target line check
            if self.active_word and self.active_word['pos'][1] > self.TARGET_LINE_Y:
                hit_reward, hit_score = self._check_hit_accuracy()
                reward += hit_reward
                self.score += hit_score
                if hit_score == 0:
                    self.missed_hits += 1
                self._spawn_next_pair()
        
        # --- Termination Check ---
        terminated = False
        if self.missed_hits >= self.MAX_MISSES:
            terminated = True
            # SFX: Game over sound
        if self.steps >= self.MAX_STEPS:
            terminated = True
            if self.track_idx >= len(self.note_track): # Completed track
                reward += 50
                # SFX: Victory sound
        
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_difficulty(self):
        if self.steps >= self.next_obstacle_increase:
            self.obstacle_frequency += 1
            self._spawn_obstacles() # Refresh obstacles with new density
            self.next_obstacle_increase += 500
        
        if self.steps >= self.next_tempo_increase:
            self.tempo = min(2.0, self.tempo + 0.05)
            self.fall_speed = min(5.0, self.fall_speed + 0.1)
            self.next_tempo_increase += 1000

    def _update_word(self, movement):
        if not self.active_word:
            return
        
        # Horizontal movement
        if movement == 3: # Left
            self.active_word['pos'][0] -= 5
        elif movement == 4: # Right
            self.active_word['pos'][0] += 5
        
        # Sizing
        if movement == 1: # Up (increase size)
            self.active_word['size'] = min(80, self.active_word['size'] + 2)
        elif movement == 2: # Down (decrease size)
            self.active_word['size'] = max(10, self.active_word['size'] - 2)

        # Boundaries
        self.active_word['pos'][0] %= self.SCREEN_WIDTH
        
        # Vertical movement (falling)
        self.active_word['pos'][1] += self.fall_speed * self.tempo
        
        # Rotation for visual flair
        self.active_word['rotation'] = (self.active_word['rotation'] + 2) % 360
        
        # Update rect for collision
        self.active_word['rect'].center = self.active_word['pos']
        self.active_word['rect'].width = self.active_word['size']
        self.active_word['rect'].height = self.active_word['size']

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1

    def _calculate_proximity_reward(self):
        if not self.active_word or not self.target_note:
            return 0.0
        
        # Reward for being horizontally aligned with the target
        dist_x = abs(self.active_word['pos'][0] - self.target_note['pos'][0])
        reward_x = max(0, 0.1 * (1 - dist_x / (self.SCREEN_WIDTH / 2)))

        # Penalty for being near an obstacle
        min_dist_y = float('inf')
        for obs in self.obstacles:
            if obs.left < self.active_word['pos'][0] < obs.right:
                dist_y = abs(self.active_word['pos'][1] - obs.centery)
                min_dist_y = min(min_dist_y, dist_y)
        
        penalty_obs = 0
        if min_dist_y < 50: # Proximity threshold
             penalty_obs = -0.1 * (1 - min_dist_y / 50)
        
        return reward_x + penalty_obs

    def _check_hit_accuracy(self):
        word = self.active_word
        note = self.target_note
        
        dist_x = abs(word['pos'][0] - note['pos'][0])
        dist_size = abs(word['size'] - note['size'])
        
        # Position accuracy (0 to 1)
        pos_accuracy = max(0, 1 - dist_x / (note['size'] * 1.5))
        # Size accuracy (0 to 1)
        size_accuracy = max(0, 1 - dist_size / (note['size'] * 1.0))
        
        total_accuracy = (pos_accuracy * 0.6 + size_accuracy * 0.4)

        if total_accuracy > 0.7: # Good Hit
            # SFX: Success chime
            self._create_particles(word['pos'], self.COLOR_PARTICLE_HIT, 30)
            return 10 * total_accuracy, 1
        else: # Miss
            # SFX: Miss/fail sound
            self._create_particles(word['pos'], self.COLOR_PARTICLE_MISS, 20)
            return -2, 0

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "missed_hits": self.missed_hits,
            "track_progress": f"{self.track_idx}/{len(self.note_track)}"
        }

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            r = self.COLOR_BG_TOP[0] + (self.COLOR_BG_BOTTOM[0] - self.COLOR_BG_TOP[0]) * y / self.SCREEN_HEIGHT
            g = self.COLOR_BG_TOP[1] + (self.COLOR_BG_BOTTOM[1] - self.COLOR_BG_TOP[1]) * y / self.SCREEN_HEIGHT
            b = self.COLOR_BG_TOP[2] + (self.COLOR_BG_BOTTOM[2] - self.COLOR_BG_TOP[2]) * y / self.SCREEN_HEIGHT
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Target Line
        pygame.draw.line(self.screen, self.COLOR_TARGET_GLOW, (0, self.TARGET_LINE_Y), (self.SCREEN_WIDTH, self.TARGET_LINE_Y), 3)
        
        # Obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs)

        # Target Note
        if self.target_note:
            pos = self.target_note['pos']
            size = self.target_note['size']
            pygame.gfxdraw.rectangle(self.screen, (int(pos[0] - size/2), int(pos[1] - size/2), size, size), self.COLOR_TARGET_GLOW)
            pygame.gfxdraw.rectangle(self.screen, (int(pos[0] - size/2), int(pos[1] - size/2), size, size), self.COLOR_TARGET)
            
        # Active Word
        if self.active_word:
            self._draw_rotated_square(
                self.screen, self.active_word['pos'], self.active_word['size'], 
                self.active_word['rotation'], self.COLOR_WORD, self.COLOR_WORD_GLOW
            )
            text_surf = self.font_word.render(str(int(self.active_word['size'])), True, self.COLOR_BG_BOTTOM)
            text_rect = text_surf.get_rect(center=self.active_word['pos'])
            self.screen.blit(text_surf, text_rect)

        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['lifetime'] / 4))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Misses
        miss_text = self.font_ui.render(f"MISSES: {self.missed_hits}/{self.MAX_MISSES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(miss_text, (self.SCREEN_WIDTH - miss_text.get_width() - 10, 10))

        # Timeline
        timeline_y = 40
        pygame.draw.rect(self.screen, self.COLOR_TIMELINE, (10, timeline_y, self.SCREEN_WIDTH - 20, 5))
        if self.note_track:
            progress = (self.track_idx / len(self.note_track))
            marker_x = 10 + progress * (self.SCREEN_WIDTH - 20)
            pygame.draw.rect(self.screen, self.COLOR_BEAT_MARKER, (marker_x - 2, timeline_y - 5, 4, 15))

    def _generate_track(self):
        self.note_track = []
        num_notes = 50
        for _ in range(num_notes):
            x = self.np_random.integers(50, self.SCREEN_WIDTH - 50)
            size = self.np_random.integers(20, 60)
            self.note_track.append({'x': x, 'size': size})
        
    def _spawn_next_pair(self):
        if self.track_idx >= len(self.note_track):
            self.active_word = None
            self.target_note = None
            return

        note_data = self.note_track[self.track_idx]
        
        self.target_note = {
            'pos': [note_data['x'], self.TARGET_LINE_Y],
            'size': note_data['size']
        }
        
        self.active_word = {
            'pos': [self.np_random.integers(50, self.SCREEN_WIDTH - 50), 50.0],
            'size': self.np_random.integers(20, 60),
            'rotation': self.np_random.uniform(0, 360),
            'rect': pygame.Rect(0, 0, 0, 0)
        }
        
        self.track_idx += 1

    def _spawn_obstacles(self):
        self.obstacles.clear()
        for _ in range(self.obstacle_frequency):
            y = self.np_random.integers(100, self.TARGET_LINE_Y - 50)
            height = 10
            width = self.np_random.integers(100, 300)
            x = self.np_random.integers(0, self.SCREEN_WIDTH - width)
            self.obstacles.append(pygame.Rect(x, y, width, height))

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'color': color,
                'lifetime': self.np_random.integers(20, 40)
            })
    
    def _draw_rotated_square(self, surface, center, size, angle, color, glow_color):
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        half_size = size / 2
        
        points = []
        for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            px = x * half_size
            py = y * half_size
            rx = px * cos_a - py * sin_a + center[0]
            ry = px * sin_a + py * cos_a + center[1]
            points.append((int(rx), int(ry)))
            
        # Glow effect
        glow_size = size + 10
        glow_half_size = glow_size / 2
        glow_points = []
        for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            px = x * glow_half_size
            py = y * glow_half_size
            rx = px * cos_a - py * sin_a + center[0]
            ry = px * sin_a + py * cos_a + center[1]
            glow_points.append((int(rx), int(ry)))
        
        pygame.gfxdraw.aapolygon(surface, glow_points, glow_color)
        pygame.gfxdraw.filled_polygon(surface, glow_points, glow_color)
        
        # Main shape
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Manual Play Example ---
    # This requires a display. Set SDL_VIDEODRIVER to something other than "dummy".
    # For example:
    # os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-comment the below to run locally with a display
    # screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    # pygame.display.set_caption("Geometric Words")
    # clock = pygame.time.Clock()
    
    # running = True
    # total_reward = 0
    
    # while running:
    #     action = [0, 0, 0] # [movement, space, shift]
        
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
        
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_UP]:
    #         action[0] = 1 # Increase size
    #     elif keys[pygame.K_DOWN]:
    #         action[0] = 2 # Decrease size
    #     elif keys[pygame.K_LEFT]:
    #         action[0] = 3 # Move left
    #     elif keys[pygame.K_RIGHT]:
    #         action[0] = 4 # Move right
    #     else:
    #         action[0] = 0 # No-op
            
    #     action[1] = 1 if keys[pygame.K_SPACE] else 0
    #     action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     total_reward += reward
        
    #     # Render the observation to the display
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     if terminated or truncated:
    #         print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    #         obs, info = env.reset()
    #         total_reward = 0
        
    #     clock.tick(30) # Run at 30 FPS
        
    # env.close()

    # The self-validation part is removed as it's not part of the final environment code
    # and can cause issues in some testing setups.
    pass