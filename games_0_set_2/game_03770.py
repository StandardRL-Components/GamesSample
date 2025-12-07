
# Generated: 2025-08-28T00:22:32.414658
# Source Brief: brief_03770.md
# Brief Index: 3770

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Press space in time with the notes to maintain speed. Avoid yellow obstacles."

    # Must be a short, user-facing description of the game:
    game_description = "A futuristic rhythm-racer. Hit the notes to the beat to go faster and complete the track."

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30 # For auto-advance, this is the step rate

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_TRACK = (40, 20, 80)
    COLOR_NEON_PINK = (255, 0, 192)
    COLOR_NEON_BLUE = (0, 221, 255)
    COLOR_NEON_GREEN = (57, 255, 20)
    COLOR_RED = (255, 20, 20)
    COLOR_YELLOW = (255, 255, 0)
    COLOR_WHITE = (240, 240, 240)

    # Game parameters
    HIT_ZONE_X = 120
    NOTE_SPAWN_X = SCREEN_WIDTH + 50
    LANE_Y_POSITIONS = [SCREEN_HEIGHT // 2 - 70, SCREEN_HEIGHT // 2, SCREEN_HEIGHT // 2 + 70]
    HIT_WINDOW = 35 # pixels

    MAX_STEPS = 5000
    MIN_ACCURACY_THRESHOLD = 0.8
    ACCURACY_GRACE_PERIOD = 20 # notes judged

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24)
            self.font_big = pygame.font.SysFont("Consolas", 48)
        except pygame.error:
            self.font_main = pygame.font.SysFont(None, 28)
            self.font_big = pygame.font.SysFont(None, 52)

        # Initialize state variables
        self.reset()
        
        # Final validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player/Rhythm state
        self.speed = 5.0
        self.base_note_speed = 5.0
        self.note_speed = self.base_note_speed
        self.notes_hit = 0
        self.notes_missed = 0
        self.combo = 0
        self.accuracy = 1.0
        
        # Input state
        self.prev_space_held = False
        
        # Entity lists
        self.notes = []
        self.obstacles = []
        self.particles = []
        
        # Procedurally generate the song track
        self._generate_song_track()
        self.track_pointer = 0

        # Background elements
        self.stars = [(self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)) for _ in range(100)]
        self.road_lines = []
        for i in range(15):
            self.road_lines.append({'x': self.np_random.integers(0, self.SCREEN_WIDTH), 'y': self.SCREEN_HEIGHT - 50 + i * 4})

        return self._get_observation(), self._get_info()

    def _generate_song_track(self):
        self.song_track = []
        current_step = 60 # Start with a small delay
        while current_step < self.MAX_STEPS - 200:
            pattern_type = self.np_random.integers(0, 10)
            if pattern_type < 7: # Single note
                lane = self.np_random.integers(0, 3)
                self.song_track.append({'step': current_step, 'type': 'note', 'lane': lane})
                current_step += self.np_random.integers(20, 40)
            elif pattern_type < 9: # Double note
                lane1 = self.np_random.integers(0, 3)
                lane2 = (lane1 + self.np_random.choice([-1, 1])) % 3
                self.song_track.append({'step': current_step, 'type': 'note', 'lane': lane1})
                self.song_track.append({'step': current_step, 'type': 'note', 'lane': lane2})
                current_step += self.np_random.integers(30, 50)
            else: # Obstacle
                lane = self.np_random.integers(0, 3)
                self.song_track.append({'step': current_step, 'type': 'obstacle', 'lane': lane})
                current_step += self.np_random.integers(40, 60)
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        # Unpack factorized action
        space_held = action[1] == 1
        space_pressed_this_frame = space_held and not self.prev_space_held
        self.prev_space_held = space_held
        
        # --- Game Logic Update ---
        self._update_entities(space_pressed_this_frame)
        reward += self._handle_collisions(space_pressed_this_frame)
        self._update_game_state()
        
        # --- Termination Check ---
        terminated = False
        total_judged = self.notes_hit + self.notes_missed
        if total_judged > self.ACCURACY_GRACE_PERIOD and self.accuracy < self.MIN_ACCURACY_THRESHOLD:
            terminated = True
            reward = -50
            self.game_over = True
            # SFX: Failure sound
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            reward = 50 if self.accuracy >= self.MIN_ACCURACY_THRESHOLD else -50
            self.game_over = True
            # SFX: Victory or Failure
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_entities(self, space_pressed):
        # Spawn new entities
        while self.track_pointer < len(self.song_track) and self.steps >= self.song_track[self.track_pointer]['step']:
            item = self.song_track[self.track_pointer]
            lane_y = self.LANE_Y_POSITIONS[item['lane']]
            if item['type'] == 'note':
                self.notes.append({'x': self.NOTE_SPAWN_X, 'y': lane_y})
            elif item['type'] == 'obstacle':
                self.obstacles.append({'x': self.NOTE_SPAWN_X, 'y': lane_y})
            self.track_pointer += 1

        # Update positions
        for entity_list in [self.notes, self.obstacles]:
            for entity in entity_list:
                entity['x'] -= self.note_speed
        
        self.particles = [p for p in self.particles if p['life'] > 1]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1

    def _handle_collisions(self, space_pressed):
        reward = 0
        # Note Hits
        if space_pressed:
            hittable_notes = [n for n in self.notes if abs(n['x'] - self.HIT_ZONE_X) < self.HIT_WINDOW]
            if hittable_notes:
                note_to_hit = min(hittable_notes, key=lambda n: abs(n['x'] - self.HIT_ZONE_X))
                self.notes.remove(note_to_hit)
                self.notes_hit += 1
                self.score += 10
                reward += 1
                self.combo += 1
                self.speed = min(15.0, self.speed + 0.5)
                if self.combo > 0 and self.combo % 5 == 0:
                    reward += 5; self.score += 50 # SFX: Combo bonus
                self._spawn_particles(self.HIT_ZONE_X, note_to_hit['y'], self.COLOR_NEON_GREEN, 20)
                # SFX: Note hit success
            else:
                self.combo = 0; self.speed = max(3.0, self.speed - 1.0) # Whiff
                self._spawn_particles(self.HIT_ZONE_X, self.SCREEN_HEIGHT/2, self.COLOR_RED, 5, spread=180)
                # SFX: Whiff sound

        # Missed Notes & Obstacles
        for note in [n for n in self.notes if n['x'] < self.HIT_ZONE_X - self.HIT_WINDOW]:
            self.notes.remove(note); self.notes_missed += 1; reward -= 1; self.score -= 5
            self.combo = 0; self.speed = max(3.0, self.speed - 1.0)
            self._spawn_particles(note['x'], note['y'], self.COLOR_RED, 10) # SFX: Note miss
        
        for obs in [o for o in self.obstacles if abs(o['x'] - self.HIT_ZONE_X) < self.HIT_WINDOW]:
            self.obstacles.remove(obs); self.notes_missed += 1; reward -= 2; self.score -= 20
            self.combo = 0; self.speed = max(2.0, self.speed - 4.0)
            self._spawn_particles(obs['x'], obs['y'], self.COLOR_YELLOW, 30, spread=360) # SFX: Obstacle hit

        # Clean up off-screen
        self.notes = [n for n in self.notes if n['x'] > -20]
        self.obstacles = [o for o in self.obstacles if o['x'] > -20]
        return reward
        
    def _update_game_state(self):
        self.speed = max(3.0, self.speed * 0.995) # Speed decay
        
        if self.steps > 0 and self.steps % 200 == 0:
            self.base_note_speed += 0.05
        self.note_speed = self.base_note_speed * (self.speed / 5.0)

        total_judged = self.notes_hit + self.notes_missed
        self.accuracy = self.notes_hit / total_judged if total_judged > 0 else 1.0
        self.score = max(0, self.score)

    def _spawn_particles(self, x, y, color, count, speed_mult=1.0, spread=90):
        for _ in range(count):
            angle = math.radians(self.np_random.uniform(-spread/2, spread/2) - 90)
            speed = self.np_random.uniform(1, 4) * speed_mult
            self.particles.append({'x': x, 'y': y, 'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed, 'life': self.np_random.integers(15, 30), 'color': color})

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_entities()
        self._render_ui()

    def _render_background(self):
        for star in self.stars:
            self.screen.set_at(star, self.COLOR_TRACK)
        for line in self.road_lines:
            line['x'] -= self.note_speed * (line['y'] / self.SCREEN_HEIGHT) * 0.8
            if line['x'] < 0: line['x'] = self.SCREEN_WIDTH
            pygame.draw.line(self.screen, self.COLOR_TRACK, (int(line['x']), line['y']), (int(line['x']) + 10, line['y']), 1)
        for y in self.LANE_Y_POSITIONS:
            pygame.draw.line(self.screen, self.COLOR_TRACK, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _render_entities(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30)); s = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, 2, 2, 2, (*p['color'], alpha))
            self.screen.blit(s, (int(p['x']-2), int(p['y']-2)))

        for note in self.notes:
            dist_factor = 1 - (note['x'] - self.HIT_ZONE_X) / self.SCREEN_WIDTH
            size = int(max(5, 20 * dist_factor)); pos = (int(note['x']), int(note['y']))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size + 3, (*self.COLOR_NEON_BLUE, 50))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_NEON_BLUE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_WHITE)

        for obs in self.obstacles:
            dist_factor = 1 - (obs['x'] - self.HIT_ZONE_X) / self.SCREEN_WIDTH
            h = w = int(max(8, 30 * dist_factor)); x, y = int(obs['x']), int(obs['y'])
            points = [(x, y-h//2), (x-w//2, y+h//2), (x+w//2, y+h//2)]
            pygame.gfxdraw.filled_trigon(self.screen, *points[0], *points[1], *points[2], (*self.COLOR_YELLOW, 80))
            pygame.gfxdraw.filled_trigon(self.screen, *points[0], *points[1], *points[2], self.COLOR_YELLOW)
            pygame.gfxdraw.aatrigon(self.screen, *points[0], *points[1], *points[2], self.COLOR_WHITE)

        ship_y = self.SCREEN_HEIGHT - 60; ship_points = [(self.HIT_ZONE_X, ship_y - 15), (self.HIT_ZONE_X - 10, ship_y + 10), (self.HIT_ZONE_X + 10, ship_y + 10)]
        pygame.gfxdraw.filled_trigon(self.screen, *ship_points[0], *ship_points[1], *ship_points[2], self.COLOR_NEON_PINK)
        pygame.gfxdraw.aatrigon(self.screen, *ship_points[0], *ship_points[1], *ship_points[2], self.COLOR_WHITE)
        
        trail_length = int(self.speed * 3); trail_y = ship_y + 12
        for i in range(trail_length):
            alpha = 200 * (1 - i / trail_length)
            pygame.draw.line(self.screen, (*self.COLOR_NEON_BLUE, int(alpha)), (self.HIT_ZONE_X - i, trail_y), (self.HIT_ZONE_X - i, trail_y), 3)

        for y in self.LANE_Y_POSITIONS:
            pygame.gfxdraw.filled_circle(self.screen, self.HIT_ZONE_X, y, 22, (*self.COLOR_NEON_PINK, 20))
            pygame.gfxdraw.aacircle(self.screen, self.HIT_ZONE_X, y, 22, (*self.COLOR_NEON_PINK, 100))

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))
        acc_text = self.font_main.render(f"ACC: {self.accuracy:.1%}", True, self.COLOR_WHITE)
        self.screen.blit(acc_text, (self.SCREEN_WIDTH - acc_text.get_width() - 10, 10))

        if self.combo > 2:
            combo_text = self.font_main.render(f"COMBO: {self.combo}", True, self.COLOR_NEON_GREEN)
            self.screen.blit(combo_text, (10, 40))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA); overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg, color = ("SONG COMPLETE", self.COLOR_NEON_GREEN) if self.accuracy >= self.MIN_ACCURACY_THRESHOLD and self.steps >= self.MAX_STEPS else ("FAILURE", self.COLOR_RED)
            end_text = self.font_big.render(msg, True, color)
            self.screen.blit(end_text, (self.SCREEN_WIDTH//2 - end_text.get_width()//2, self.SCREEN_HEIGHT//2 - end_text.get_height()//2 - 20))
            final_score_text = self.font_main.render(f"Final Score: {self.score}", True, self.COLOR_WHITE)
            self.screen.blit(final_score_text, (self.SCREEN_WIDTH//2 - final_score_text.get_width()//2, self.SCREEN_HEIGHT//2 + 20))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "accuracy": self.accuracy, "combo": self.combo}
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,) and self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3) and test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3) and isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3) and isinstance(reward, (int, float))
        assert isinstance(term, bool) and not trunc and isinstance(info, dict)
        print("✓ Implementation validated successfully")

# Example of how to run the environment for interactive play
if __name__ == '__main__':
    import time

    class HumanGameEnv(GameEnv):
        metadata = {"render_modes": ["rgb_array", "human"], "render_fps": GameEnv.FPS}
        def __init__(self, render_mode="human"):
            super().__init__(render_mode=render_mode)
            self.render_mode = render_mode
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
                pygame.display.set_caption("Rhythm Racer")

        def _get_observation(self):
            obs = super()._get_observation()
            if self.render_mode == "human": self.window.blit(self.screen, (0, 0)); pygame.display.flip()
            return obs

        def close(self):
            if hasattr(self, 'window'): pygame.display.quit()
            super().close()

    env = HumanGameEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    terminated, quit_game = False, False
    action = env.action_space.sample(); action.fill(0)
    
    print(f"\n{'='*30}\n      RHYTHM RACER DEMO\n{'='*30}\n{GameEnv.user_guide}\nClose the window to quit.\n{'='*30}\n")

    while not terminated and not quit_game:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: quit_game = True
        
        keys = pygame.key.get_pressed()
        action[0] = 1 if keys[pygame.K_UP] else 2 if keys[pygame.K_DOWN] else 3 if keys[pygame.K_LEFT] else 4 if keys[pygame.K_RIGHT] else 0
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.clock.tick(env.FPS)

    print(f"\n{'='*30}\n       GAME OVER\nFinal Score: {info['score']}\nFinal Accuracy: {info['accuracy']:.1%}\n{'='*30}\n")
    time.sleep(3)
    env.close()