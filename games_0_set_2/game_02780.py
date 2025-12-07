
# Generated: 2025-08-27T21:24:53.662383
# Source Brief: brief_02780.md
# Brief Index: 2780

        
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
        "Controls: ←→ to move the cursor. Press space to hit notes on the beat line."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced rhythm game. Move your cursor and hit the falling notes "
        "on the beat line to score points and build your accuracy."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    NUM_COLUMNS = 4
    NUM_NOTES = 100
    MAX_STEPS = 3000

    # Colors (Neon Theme)
    COLOR_BG = (10, 5, 25)
    COLOR_COLUMN = (20, 10, 50, 100)
    COLOR_BEAT_LINE = (0, 255, 255)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_NOTE = (255, 0, 128)
    COLOR_TEXT = (220, 220, 255)
    COLOR_PERFECT = (0, 255, 0)
    COLOR_GOOD = (255, 255, 0)
    COLOR_MISS = (255, 50, 50)

    # Gameplay Constants
    BEAT_LINE_Y = 340
    PERFECT_TOLERANCE = 8
    GOOD_TOLERANCE = 20
    MOVE_COOLDOWN = 4  # frames

    # Reward Constants
    REWARD_PERFECT = 1.0
    REWARD_GOOD = 0.5
    REWARD_MISS = -1.0
    REWARD_ACCURACY_BONUS = 5.0
    REWARD_WIN = 50.0
    REWARD_FAIL = -50.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_feedback = pygame.font.SysFont("sans", 32, bold=True)
        self.font_gameover = pygame.font.SysFont("sans", 48, bold=True)

        # Layout calculation
        self.COLUMN_WIDTH = self.SCREEN_WIDTH // 8
        self.LANE_AREA_WIDTH = self.COLUMN_WIDTH * self.NUM_COLUMNS
        self.LANE_START_X = (self.SCREEN_WIDTH - self.LANE_AREA_WIDTH) // 2
        self.NOTE_WIDTH = int(self.COLUMN_WIDTH * 0.8)

        # Initialize state variables
        self.cursor_pos = 0
        self.notes = []
        self.song_data = []
        self.particles = []
        self.feedback_messages = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_over_message = ""
        self.note_speed = 0.0
        self.hits_perfect = 0
        self.hits_good = 0
        self.misses = 0
        self.total_notes_processed = 0
        self.prev_space_held = False
        self.accuracy_bonus_awarded = False
        self.move_timer = 0
        
        self.reset()
        
        # self.validate_implementation() # Optional self-check

    def _generate_song(self):
        self.song_data.clear()
        current_step = 60  # Give player some time to prepare
        for _ in range(self.NUM_NOTES):
            # Time between notes: 0.5s to 1.5s at 30fps
            step_interval = self.np_random.integers(15, 45)
            current_step += step_interval
            column = self.np_random.integers(0, self.NUM_COLUMNS)
            self.song_data.append({'spawn_step': current_step, 'column': column})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_song()

        self.cursor_pos = self.NUM_COLUMNS // 2
        self.notes = []
        self.particles = []
        self.feedback_messages = []
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_over_message = ""
        
        self.note_speed = 2.0
        self.hits_perfect = 0
        self.hits_good = 0
        self.misses = 0
        self.total_notes_processed = 0
        
        self.prev_space_held = False
        self.accuracy_bonus_awarded = False
        self.move_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        
        # --- Action Handling ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held
        
        if self.move_timer > 0:
            self.move_timer -= 1

        if self.move_timer <= 0:
            if movement == 3:  # Left
                self.cursor_pos = max(0, self.cursor_pos - 1)
                self.move_timer = self.MOVE_COOLDOWN
            elif movement == 4:  # Right
                self.cursor_pos = min(self.NUM_COLUMNS - 1, self.cursor_pos + 1)
                self.move_timer = self.MOVE_COOLDOWN

        # --- Game Logic ---
        self.steps += 1
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.note_speed = min(8.0, self.note_speed + 0.02)

        # Spawn new notes
        self.song_data = [note for note in self.song_data if not (
            note['spawn_step'] == self.steps and self.notes.append(
                {'column': note['column'], 'y': -self.NOTE_WIDTH, 'hit': False}
            )
        )]

        # Move existing notes
        for note in self.notes:
            note['y'] += self.note_speed

        # Hit detection on space press
        if space_pressed:
            # sfx: Hit attempt sound
            notes_in_lane = [n for n in self.notes if n['column'] == self.cursor_pos and not n['hit']]
            if notes_in_lane:
                # Find the note closest to the beat line
                closest_note = min(notes_in_lane, key=lambda n: abs(n['y'] - self.BEAT_LINE_Y))
                dist = abs(closest_note['y'] - self.BEAT_LINE_Y)
                hit_pos = (self._get_column_center_x(self.cursor_pos), self.BEAT_LINE_Y)

                if dist <= self.PERFECT_TOLERANCE:
                    reward += self.REWARD_PERFECT
                    self.hits_perfect += 1
                    closest_note['hit'] = True
                    self._create_feedback("Perfect!", self.COLOR_PERFECT, hit_pos)
                    self._create_particles(hit_pos, self.COLOR_PERFECT, 30)
                    # sfx: Perfect hit
                elif dist <= self.GOOD_TOLERANCE:
                    reward += self.REWARD_GOOD
                    self.hits_good += 1
                    closest_note['hit'] = True
                    self._create_feedback("Good", self.COLOR_GOOD, hit_pos)
                    self._create_particles(hit_pos, self.COLOR_GOOD, 20)
                    # sfx: Good hit
        
        # Process misses and remove old notes
        notes_to_remove = []
        for note in self.notes:
            if note['hit']:
                self.total_notes_processed += 1
                notes_to_remove.append(note)
            elif note['y'] > self.BEAT_LINE_Y + self.GOOD_TOLERANCE:
                reward += self.REWARD_MISS
                self.misses += 1
                self.total_notes_processed += 1
                notes_to_remove.append(note)
                hit_pos = (self._get_column_center_x(note['column']), self.BEAT_LINE_Y)
                self._create_feedback("Miss", self.COLOR_MISS, hit_pos)
                # sfx: Miss sound
        self.notes = [n for n in self.notes if n not in notes_to_remove]

        # Update particles and feedback messages
        self._update_effects()
        
        # Accuracy bonus
        accuracy = self._get_accuracy()
        if not self.accuracy_bonus_awarded and accuracy >= 0.8 and self.total_notes_processed > 10:
            reward += self.REWARD_ACCURACY_BONUS
            self.accuracy_bonus_awarded = True

        self.score += reward

        # --- Termination Check ---
        terminated = False
        if self.total_notes_processed > 10 and accuracy < 0.5:
            terminated = True
            self.score += self.REWARD_FAIL
            self.game_over_message = "Failure: Accuracy < 50%"
        elif self.total_notes_processed >= self.NUM_NOTES:
            terminated = True
            if accuracy >= 0.8:
                self.score += self.REWARD_WIN
                self.game_over_message = "Song Cleared!"
            else:
                self.score += self.REWARD_FAIL
                self.game_over_message = f"Finished: {accuracy:.1%} Accuracy"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.score += self.REWARD_FAIL
            self.game_over_message = "Time Out!"
        
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "accuracy": self._get_accuracy(),
            "notes_processed": self.total_notes_processed,
        }

    def _get_accuracy(self):
        total_hits = self.hits_perfect + self.hits_good
        if self.total_notes_processed == 0:
            return 1.0
        return total_hits / self.total_notes_processed
    
    def _get_column_center_x(self, column_index):
        return self.LANE_START_X + (column_index * self.COLUMN_WIDTH) + (self.COLUMN_WIDTH // 2)

    def _render_game(self):
        # Draw columns
        for i in range(self.NUM_COLUMNS):
            col_rect = pygame.Rect(self.LANE_START_X + i * self.COLUMN_WIDTH, 0, self.COLUMN_WIDTH, self.SCREEN_HEIGHT)
            s = pygame.Surface((self.COLUMN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill(self.COLOR_COLUMN)
            self.screen.blit(s, col_rect.topleft)

        # Draw beat line with glow
        for i in range(5):
            alpha = 150 - i * 30
            pygame.gfxdraw.hline(self.screen, 0, self.SCREEN_WIDTH, self.BEAT_LINE_Y - i, (*self.COLOR_BEAT_LINE, alpha))
            pygame.gfxdraw.hline(self.screen, 0, self.SCREEN_WIDTH, self.BEAT_LINE_Y + i, (*self.COLOR_BEAT_LINE, alpha))
        pygame.draw.line(self.screen, self.COLOR_BEAT_LINE, (0, self.BEAT_LINE_Y), (self.SCREEN_WIDTH, self.BEAT_LINE_Y), 2)
        
        # Draw notes
        for note in self.notes:
            note_x = self._get_column_center_x(note['column'])
            note_rect = pygame.Rect(note_x - self.NOTE_WIDTH // 2, int(note['y']) - self.NOTE_WIDTH // 2, self.NOTE_WIDTH, self.NOTE_WIDTH)
            pygame.draw.rect(self.screen, self.COLOR_NOTE, note_rect, border_radius=4)
            pygame.draw.rect(self.screen, (255,255,255), note_rect, 2, border_radius=4)

        # Draw cursor
        cursor_x = self._get_column_center_x(self.cursor_pos)
        cursor_rect = pygame.Rect(cursor_x - self.NOTE_WIDTH // 2, self.BEAT_LINE_Y - self.NOTE_WIDTH // 2, self.NOTE_WIDTH, self.NOTE_WIDTH)
        # Glow
        glow_rect = cursor_rect.inflate(10, 10)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (*self.COLOR_CURSOR, 50), s.get_rect(), border_radius=8)
        self.screen.blit(s, glow_rect.topleft)
        # Main cursor
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, border_radius=6)

        # Draw particles and feedback
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['radius']))
        for msg in self.feedback_messages:
            self.screen.blit(msg['surface'], msg['pos'])

    def _render_ui(self):
        # Accuracy
        accuracy_text = f"Accuracy: {self._get_accuracy():.1%}"
        acc_surf = self.font_main.render(accuracy_text, True, self.COLOR_TEXT)
        self.screen.blit(acc_surf, (10, 10))

        # Score
        score_text = f"Score: {int(self.score)}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10))

        # Song Progress
        progress = self.total_notes_processed / self.NUM_NOTES
        bar_width = 200
        bar_height = 10
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = 20
        pygame.draw.rect(self.screen, self.COLOR_COLUMN, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BEAT_LINE, (bar_x, bar_y, bar_width * progress, bar_height), border_radius=5)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        text_surf = self.font_gameover.render(self.game_over_message, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = self.np_random.uniform(2, 5)
            life = self.np_random.integers(15, 30) # frames
            self.particles.append({'pos': list(pos), 'vel': vel, 'radius': radius, 'life': life, 'color': color})

    def _create_feedback(self, text, color, pos):
        surface = self.font_feedback.render(text, True, color)
        rect = surface.get_rect(center=pos)
        self.feedback_messages.append({'surface': surface, 'pos': rect.topleft, 'life': 45, 'vel_y': -0.5})

    def _update_effects(self):
        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['radius'] *= 0.95
            p['life'] -= 1
            if p['life'] <= 0 or p['radius'] < 0.5:
                self.particles.remove(p)
        
        # Update feedback messages
        for msg in self.feedback_messages[:]:
            msg['pos'] = (msg['pos'][0], msg['pos'][1] + msg['vel_y'])
            msg['life'] -= 1
            alpha = max(0, min(255, int(255 * (msg['life'] / 30))))
            msg['surface'].set_alpha(alpha)
            if msg['life'] <= 0:
                self.feedback_messages.remove(msg)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Rhythm Grid Racer")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    done = False
    while not done:
        # --- Human Controls ---
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        elif keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Accuracy: {info['accuracy']:.1%}")
            # Wait for a moment before resetting
            pygame.time.wait(3000)
            obs, info = env.reset()

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Frame Rate ---
        clock.tick(30) # Run at 30 FPS

    env.close()