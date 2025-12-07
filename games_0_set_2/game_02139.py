
# Generated: 2025-08-28T03:50:21.720151
# Source Brief: brief_02139.md
# Brief Index: 2139

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Press space to slice the falling notes as they cross the line."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced rhythm game. Slice the falling musical notes in time to complete the song and maximize your score."
    )

    # Frames auto-advance for this real-time rhythm game.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_STEPS = 6000
        self.MAX_MISSES = 3
        self.NUM_LANES = 5
        self.NUM_NOTES = 150

        # Gameplay constants
        self.SLICE_LINE_Y = 320
        self.NOTE_HEIGHT = 20
        self.NOTE_WIDTH = 50
        self.SLICE_TOLERANCE = 15  # Hitbox height for slicing
        self.INITIAL_NOTE_SPEED = 4.0
        self.DIFFICULTY_INCREASE = 1.5

        # Visual constants
        self.COLOR_BG_TOP = (15, 20, 45)
        self.COLOR_BG_BOTTOM = (30, 10, 35)
        self.COLOR_SLICE_LINE = (0, 255, 255)
        self.COLOR_SLICE_GLOW = (0, 150, 150)
        self.COLOR_TEXT = (240, 240, 255)
        self.COLOR_MISS = (255, 50, 50)
        self.NOTE_COLORS = [
            (255, 0, 128),   # Magenta
            (255, 255, 0),   # Yellow
            (0, 255, 0),     # Green
            (255, 128, 0),   # Orange
            (0, 128, 255),   # Blue
        ]

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Lane positions
        self.lanes_x = [int(self.SCREEN_WIDTH * (i + 1) / (self.NUM_LANES + 1)) for i in range(self.NUM_LANES)]

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.note_speed = self.INITIAL_NOTE_SPEED
        self.notes = []
        self.particles = []
        self.miss_indicators = []
        self.song_data = []
        self.song_index = 0
        self.last_space_state = False
        self.slice_flash_timer = 0
        
        # Must be called at the end of __init__
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.note_speed = self.INITIAL_NOTE_SPEED
        self.notes = []
        self.particles = []
        self.miss_indicators = []
        self.last_space_state = False
        self.slice_flash_timer = 0
        
        # Generate the "song"
        self.song_data = []
        current_step = 100 # Start notes after a brief delay
        for _ in range(self.NUM_NOTES):
            lane = self.np_random.integers(0, self.NUM_LANES)
            step_increment = self.np_random.integers(20, 50)
            current_step += step_increment
            if current_step < self.MAX_STEPS - 200:
                self.song_data.append({'step': current_step, 'lane': lane})
        self.song_index = 0

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]  # unused
        space_held = action[1] == 1
        shift_held = action[2] == 1  # unused

        self.steps += 1
        reward = 0
        
        # --- Game Logic ---
        self._update_difficulty()
        self._spawn_notes()
        self._update_notes()
        
        # Detect slice action on key press (rising edge)
        slice_attempt = space_held and not self.last_space_state
        if slice_attempt:
            self.slice_flash_timer = 5 # Flash for 5 frames
            # sfx: Slice sound
        
        notes_sliced_this_frame = self._handle_slices(slice_attempt)
        
        if notes_sliced_this_frame > 0:
            reward += notes_sliced_this_frame * 1.0
        else:
            reward -= 0.2 # Penalty for inaction/mistimed slice
        
        self._update_particles()
        self._update_miss_indicators()
        
        self.last_space_state = space_held
        
        # --- Termination ---
        terminated = self.misses >= self.MAX_MISSES or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.misses < self.MAX_MISSES and self.steps >= self.MAX_STEPS:
                reward += 100 # Victory bonus
                
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _spawn_notes(self):
        while self.song_index < len(self.song_data) and self.steps >= self.song_data[self.song_index]['step']:
            note_data = self.song_data[self.song_index]
            lane = note_data['lane']
            new_note = {
                'x': self.lanes_x[lane],
                'y': -self.NOTE_HEIGHT,
                'color': self.NOTE_COLORS[lane],
            }
            self.notes.append(new_note)
            self.song_index += 1

    def _update_difficulty(self):
        if self.steps == 2000 or self.steps == 4000:
            self.note_speed += self.DIFFICULTY_INCREASE

    def _update_notes(self):
        notes_to_remove = []
        for note in self.notes:
            note['y'] += self.note_speed
            if note['y'] > self.SCREEN_HEIGHT:
                self.misses += 1
                notes_to_remove.append(note)
                self._create_miss_indicator(note['x'])
                # sfx: Miss sound
        
        self.notes = [n for n in self.notes if n not in notes_to_remove]

    def _handle_slices(self, slice_attempt):
        notes_hit = 0
        if not slice_attempt:
            return notes_hit

        notes_to_remove = []
        for note in self.notes:
            note_center_y = note['y'] + self.NOTE_HEIGHT / 2
            if abs(note_center_y - self.SLICE_LINE_Y) < self.SLICE_TOLERANCE:
                notes_to_remove.append(note)
                self.score += 1
                notes_hit += 1
                self._create_particles(note['x'], self.SLICE_LINE_Y, note['color'])
                # sfx: Hit sound
        
        if notes_hit > 0:
            self.notes = [n for n in self.notes if n not in notes_to_remove]
        
        return notes_hit
        
    def _create_particles(self, x, y, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 5)
            particle = {
                'x': x,
                'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'lifespan': self.np_random.integers(15, 30),
                'color': color,
                'size': self.np_random.integers(2, 5)
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        
    def _create_miss_indicator(self, x):
        self.miss_indicators.append({'x': x, 'y': self.SLICE_LINE_Y, 'lifespan': 30})

    def _update_miss_indicators(self):
        for m in self.miss_indicators:
            m['lifespan'] -= 1
        self.miss_indicators = [m for m in self.miss_indicators if m['lifespan'] > 0]

    def _get_observation(self):
        # Draw background gradient
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))
        
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw slice line
        if self.slice_flash_timer > 0:
            self.slice_flash_timer -= 1
            pygame.draw.line(self.screen, (255, 255, 255), (0, self.SLICE_LINE_Y), (self.SCREEN_WIDTH, self.SLICE_LINE_Y), 5)
        else:
            pygame.draw.line(self.screen, self.COLOR_SLICE_GLOW, (0, self.SLICE_LINE_Y), (self.SCREEN_WIDTH, self.SLICE_LINE_Y), 3)
            pygame.draw.line(self.screen, self.COLOR_SLICE_LINE, (0, self.SLICE_LINE_Y), (self.SCREEN_WIDTH, self.SLICE_LINE_Y), 1)

        # Draw notes
        for note in self.notes:
            rect = pygame.Rect(note['x'] - self.NOTE_WIDTH / 2, note['y'], self.NOTE_WIDTH, self.NOTE_HEIGHT)
            pygame.draw.rect(self.screen, note['color'], rect, border_radius=4)
            pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in note['color']), rect, 2, border_radius=4)
            
        # Draw miss indicators
        for m in self.miss_indicators:
            alpha = int(255 * (m['lifespan'] / 30))
            color = (*self.COLOR_MISS, alpha)
            x, y = int(m['x']), int(m['y'])
            size = 20
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.line(s, color, (0, 0), (size, size), 4)
            pygame.draw.line(s, color, (size, 0), (0, size), 4)
            self.screen.blit(s, (x - size/2, y - size/2))
            
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30))
            color = (*p['color'], alpha)
            pos = (int(p['x']), int(p['y']))
            size = int(p['size'] * (p['lifespan'] / 30))
            # Using gfxdraw for anti-aliased circles
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], max(0, size), color)


    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Misses
        miss_text = self.font_small.render("Misses:", True, self.COLOR_TEXT)
        self.screen.blit(miss_text, (self.SCREEN_WIDTH - 150, 15))
        for i in range(self.MAX_MISSES):
            color = self.COLOR_MISS if i < self.misses else (80, 80, 80)
            pygame.draw.circle(self.screen, color, (self.SCREEN_WIDTH - 50 + i * 25, 25), 8)

        # Progress bar
        progress = self.steps / self.MAX_STEPS
        bar_width = self.SCREEN_WIDTH - 40
        pygame.draw.rect(self.screen, (50, 50, 80), (20, self.SCREEN_HEIGHT - 25, bar_width, 15), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_SLICE_LINE, (20, self.SCREEN_HEIGHT - 25, bar_width * progress, 15), border_radius=4)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "SONG CLEAR" if self.misses < self.MAX_MISSES else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misses": self.misses,
        }
        
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# Example usage:
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Setup a window to view the game
    pygame.display.set_caption("Rhythm Slicer")
    real_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    # Game loop
    running = True
    while running:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get keyboard state
        keys = pygame.key.get_pressed()
        space_pressed = keys[pygame.K_SPACE]
        
        # Construct the action from keyboard input
        action = env.action_space.null_action()
        action[1] = 1 if space_pressed else 0
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(30)
        
    env.close()