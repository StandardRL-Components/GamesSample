import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:50:09.006549
# Source Brief: brief_02407.md
# Brief Index: 2407
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
        "A rhythm-based puzzle game where you repair DNA strands. Match the directional "
        "sequence to the beat to score points and advance."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to match the sequence displayed at the top "
        "of the screen. Time your inputs with the beat for a higher score."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
    FPS = 30  # Assumed FPS for smooth interpolation

    # Colors (Bright, High Contrast)
    COLOR_BG = (10, 15, 30)
    COLOR_GRID = (25, 35, 60)
    COLOR_STRAND = (40, 60, 120)
    COLOR_BASE = (80, 100, 180)
    COLOR_PLAYER = (0, 255, 255) # Bright Cyan
    COLOR_PLAYER_GLOW = (0, 150, 255)
    COLOR_TARGET_UI = (255, 255, 0) # Yellow
    COLOR_CORRECT = (0, 255, 128) # Bright Green
    COLOR_INCORRECT = (255, 50, 50) # Bright Red
    COLOR_TEXT = (220, 220, 240)
    COLOR_TIME_BAR = (255, 180, 0)

    # Game Mechanics
    INITIAL_BPM = 60.0
    INITIAL_SEQUENCE_LENGTH = 4
    MAX_STEPS = 5000
    TIMING_WINDOW_MS = 150 # Generous timing window for player
    STRAND_LENGTH = 180

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("consola", 24, bold=True)
        self.font_small = pygame.font.SysFont("consola", 16)
        self.font_icon = pygame.font.SysFont("segoeuisymbol", 28)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.bpm = 0.0
        self.beat_duration_ms = 0.0
        self.last_beat_time = 0.0
        self.sequence_length = 0
        self.time_limit_ms = 0
        self.start_time_ms = 0
        self.successful_sequences = 0
        self.target_sequence = []
        self.current_sequence_index = 0
        self.particles = []
        self.action_feedback = [] # For rendering flashes/beams
        
        # self.reset() # reset is called by the environment wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.bpm = self.INITIAL_BPM
        self.beat_duration_ms = 60000.0 / self.bpm
        self.last_beat_time = pygame.time.get_ticks()
        self.start_time_ms = pygame.time.get_ticks()
        
        self.sequence_length = self.INITIAL_SEQUENCE_LENGTH
        self.time_limit_ms = 15 * 1000
        self.successful_sequences = 0
        
        self.particles = []
        self.action_feedback = []
        
        self._generate_new_sequence()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        current_time_ms = pygame.time.get_ticks()

        # --- Unpack Action ---
        movement = action[0]  # 0=none, 1=up, 2=down, 3=left, 4=right

        # --- Beat and Timing Logic ---
        time_since_beat = current_time_ms - self.last_beat_time
        if time_since_beat >= self.beat_duration_ms:
            self.last_beat_time += self.beat_duration_ms
            time_since_beat = current_time_ms - self.last_beat_time
        
        is_in_window = abs(time_since_beat) < self.TIMING_WINDOW_MS or \
                       abs(time_since_beat - self.beat_duration_ms) < self.TIMING_WINDOW_MS

        # --- Handle Player Action ---
        if movement != 0:
            target_action = self.target_sequence[self.current_sequence_index]
            
            if is_in_window:
                reward += 1  # Reward for good timing
                if movement == target_action:
                    # --- Correct Action ---
                    # // SFX: Correct_Teleport.wav
                    reward += 10
                    self.score += 10
                    self._create_action_feedback('correct', movement, current_time_ms)
                    self.current_sequence_index += 1
                    
                    if self.current_sequence_index >= self.sequence_length:
                        # --- Sequence Complete ---
                        # // SFX: Sequence_Complete.wav
                        reward += 100
                        self.score += 100
                        self.successful_sequences += 1
                        
                        if self.successful_sequences > 0 and self.successful_sequences % 5 == 0:
                            self.bpm += 2.0
                            self.beat_duration_ms = 60000.0 / self.bpm
                            self.sequence_length += 1
                        
                        self._generate_new_sequence()
                        self.start_time_ms = current_time_ms
                        self.time_limit_ms = (8 + self.sequence_length) * 1000 # More time for longer seq
                else:
                    # --- Incorrect Action ---
                    # // SFX: Failure.wav
                    reward -= 5
                    self.score -= 5
                    self._create_action_feedback('incorrect', movement, current_time_ms)
                    self.game_over = True
            else:
                # --- Mistimed Action ---
                # // SFX: Mistimed_Hit.wav
                reward -= 1
                self.score -= 1
                self._create_action_feedback('mistimed', movement, current_time_ms)
        
        # --- Update Game State ---
        self._update_effects(current_time_ms)
        terminated = self._check_termination(current_time_ms)
        
        # Apply terminal rewards
        if terminated and not self.game_over: # Time ran out
             # // SFX: Time_Out.wav
            reward -= 100
            self.score -= 100
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _check_termination(self, current_time_ms):
        if self.game_over:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        time_elapsed = current_time_ms - self.start_time_ms
        if time_elapsed >= self.time_limit_ms:
            return True
        return False

    def _get_observation(self):
        # --- Clear Screen ---
        self.screen.fill(self.COLOR_BG)
        
        # --- Render All Game Elements ---
        self._render_background()
        self._render_dna_strands()
        self._render_effects()
        self._render_player()
        self._render_ui()
        
        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bpm": self.bpm,
            "sequence_length": self.sequence_length,
        }

    # --- Helper and Rendering Methods ---

    def _generate_new_sequence(self):
        self.target_sequence = [
            self.np_random.integers(1, 5) for _ in range(self.sequence_length)
        ]
        self.current_sequence_index = 0

    def _map_direction_to_vector(self, direction):
        if direction == 1: return (0, -1) # Up
        if direction == 2: return (0, 1)  # Down
        if direction == 3: return (-1, 0) # Left
        if direction == 4: return (1, 0)  # Right
        return (0, 0)

    def _update_effects(self, current_time_ms):
        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
        
        # Update action feedback (fade out)
        for fb in self.action_feedback[:]:
            if current_time_ms - fb['start_time'] > fb['duration']:
                self.action_feedback.remove(fb)

    def _create_action_feedback(self, type, direction, current_time_ms):
        color_map = {
            'correct': self.COLOR_CORRECT,
            'incorrect': self.COLOR_INCORRECT,
            'mistimed': self.COLOR_PLAYER_GLOW
        }
        feedback = {
            'type': type,
            'direction': direction,
            'start_time': current_time_ms,
            'duration': 400,
            'color': color_map[type]
        }
        self.action_feedback.append(feedback)
        
        if type == 'correct':
            dir_vec = self._map_direction_to_vector(direction)
            end_point = (self.CENTER_X + dir_vec[0] * self.STRAND_LENGTH, 
                         self.CENTER_Y + dir_vec[1] * self.STRAND_LENGTH)
            for _ in range(30):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                self.particles.append({
                    'pos': list(end_point),
                    'vel': vel,
                    'life': self.np_random.integers(15, 30),
                    'color': self.COLOR_CORRECT,
                    'size': self.np_random.uniform(1, 4)
                })

    def _render_background(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_dna_strands(self):
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)] # U, D, L, R
        for d_idx, d in enumerate(directions):
            start_pos = (self.CENTER_X, self.CENTER_Y)
            end_pos = (self.CENTER_X + d[0] * self.STRAND_LENGTH, 
                       self.CENTER_Y + d[1] * self.STRAND_LENGTH)
            
            # Draw main strand line
            pygame.draw.aaline(self.screen, self.COLOR_STRAND, start_pos, end_pos, 2)
            
            # Draw bases
            for i in range(1, 6):
                p = i / 5.0
                pos = (start_pos[0] * (1-p) + end_pos[0] * p, 
                       start_pos[1] * (1-p) + end_pos[1] * p)
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 4, self.COLOR_BASE)
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 4, self.COLOR_STRAND)
                
            # Highlight target base for current step
            if not self.game_over and d_idx + 1 == self.target_sequence[self.current_sequence_index]:
                time_since_beat = pygame.time.get_ticks() - self.last_beat_time
                beat_progress = time_since_beat / self.beat_duration_ms
                pulse_alpha = max(0, 255 * (1.0 - beat_progress))
                
                color = self.COLOR_TARGET_UI + (int(pulse_alpha),)
                pygame.gfxdraw.filled_circle(self.screen, int(end_pos[0]), int(end_pos[1]), 10, color)

    def _render_player(self):
        # Pulsing glow based on beat
        time_since_beat = pygame.time.get_ticks() - self.last_beat_time
        beat_progress = min(1.0, time_since_beat / self.beat_duration_ms)
        pulse = (1.0 - beat_progress) ** 2
        
        glow_radius = int(15 + 10 * pulse)
        glow_alpha = int(50 + 100 * pulse)
        glow_color = self.COLOR_PLAYER_GLOW + (glow_alpha,)
        pygame.gfxdraw.filled_circle(self.screen, self.CENTER_X, self.CENTER_Y, glow_radius, glow_color)
        
        # Core nanobot
        pygame.gfxdraw.filled_circle(self.screen, self.CENTER_X, self.CENTER_Y, 12, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, self.CENTER_X, self.CENTER_Y, 12, (200, 255, 255))
        
    def _render_effects(self):
        current_time_ms = pygame.time.get_ticks()
        
        # Render particles
        for p in self.particles:
            size = p['size'] * (p['life'] / 30.0)
            if size > 1:
                pygame.draw.circle(self.screen, p['color'], p['pos'], size)

        # Render action feedback (beams and flashes)
        for fb in self.action_feedback:
            progress = (current_time_ms - fb['start_time']) / fb['duration']
            alpha = int(255 * (1.0 - progress**2))
            
            if fb['type'] == 'correct' or fb['type'] == 'mistimed':
                dir_vec = self._map_direction_to_vector(fb['direction'])
                end_pos = (self.CENTER_X + dir_vec[0] * self.STRAND_LENGTH, 
                           self.CENTER_Y + dir_vec[1] * self.STRAND_LENGTH)
                color = fb['color']
                
                # Draw a fading beam
                if alpha > 0:
                    try:
                        pygame.draw.line(self.screen, color + (alpha,), 
                                         (self.CENTER_X, self.CENTER_Y), end_pos, width=int(5 * (1-progress)))
                    except TypeError: # Color might not have alpha
                        pygame.draw.line(self.screen, tuple(list(color)[:3]) + (alpha,), 
                                         (self.CENTER_X, self.CENTER_Y), end_pos, width=int(5 * (1-progress)))


            elif fb['type'] == 'incorrect':
                # Flash the whole screen red
                s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                try:
                    s.fill(fb['color'] + (int(alpha * 0.5),))
                except TypeError: # Color might not have alpha
                    s.fill(tuple(list(fb['color'])[:3]) + (int(alpha * 0.5),))
                self.screen.blit(s, (0, 0))

    def _render_ui(self):
        # --- Score ---
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # --- Time Bar ---
        time_elapsed = pygame.time.get_ticks() - self.start_time_ms
        time_ratio = max(0, 1.0 - (time_elapsed / self.time_limit_ms))
        bar_width = (self.WIDTH - 20) * time_ratio
        pygame.draw.rect(self.screen, self.COLOR_TIME_BAR, (10, self.HEIGHT - 20, bar_width, 10))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (10, self.HEIGHT - 20, self.WIDTH - 20, 10), 1)

        # --- Target Sequence Display ---
        arrow_map = {1: '↑', 2: '↓', 3: '←', 4: '→'}
        seq_width = self.sequence_length * 35
        start_x = self.CENTER_X - seq_width // 2
        
        for i, action_id in enumerate(self.target_sequence):
            char = arrow_map.get(action_id, '?')
            color = self.COLOR_GRID
            if i < self.current_sequence_index:
                color = self.COLOR_CORRECT
            elif i == self.current_sequence_index:
                color = self.COLOR_TARGET_UI
            
            arrow_surf = self.font_icon.render(char, True, color)
            rect = arrow_surf.get_rect(center=(start_x + i * 35, 45))
            self.screen.blit(arrow_surf, rect)

            if i == self.current_sequence_index:
                pygame.draw.rect(self.screen, self.COLOR_TARGET_UI, rect.inflate(10, 5), 1)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually
    # For manual play, we need a display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for manual play
    pygame.display.set_caption("DNA Repair Rhythm Game")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default action: do nothing
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # In manual play, we only trigger an action on a key press.
        # For an agent, it would provide an action every step.
        # This loop simulates taking one action then coasting.
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
            if terminated or truncated:
                print("--- Episode Finished --- (Press 'R' to reset)")
        else: # If no key is pressed, just advance the environment state
            obs, reward, terminated, truncated, info = env.step([0,0,0])
            if terminated or truncated:
                if not env.game_over: # Prevent printing on every frame after termination
                    print("--- Episode Finished --- (Press 'R' to reset)")


        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)
        
    env.close()