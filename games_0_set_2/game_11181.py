import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:37:24.309060
# Source Brief: brief_01181.md
# Brief Index: 1181
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
        "Catch falling numbers with your cursor. Create combos by capturing multiple numbers at "
        "once to maximize your score before time runs out."
    )
    user_guide = "Use the arrow keys (↑↓←→) to move your cursor and catch the falling numbers."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    
    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_CURSOR = (0, 255, 128)
    COLOR_CURSOR_GLOW = (0, 255, 128, 50)
    COLOR_NUMBER = (255, 255, 255)
    COLOR_PARTICLE = (255, 220, 0)
    COLOR_UI = (220, 220, 220)
    COLOR_COMBO = (255, 220, 0)

    # Game Parameters
    CURSOR_SIZE = 20
    CURSOR_SPEED = 8.0
    INITIAL_SPAWN_INTERVAL = 3.0  # seconds
    MIN_SPAWN_INTERVAL = 1.0  # seconds
    DIFFICULTY_INTERVAL = 30.0 # seconds
    SPAWN_RATE_INCREASE = 0.1 # seconds
    
    NUMBER_START_VEL = 1.0
    NUMBER_ACCELERATION = 0.02
    
    MAX_TIME = 120.0  # seconds
    WIN_SCORE = 1000
    MAX_STEPS = 10000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_combo = pygame.font.SysFont("monospace", 32, bold=True)
        
        # State Variables (initialized in reset)
        self.cursor_pos = None
        self.score = None
        self.time_remaining = None
        self.numbers = None
        self.particles = None
        self.combo_texts = None
        self.spawn_timer = None
        self.spawn_interval = None
        self.difficulty_timer = None
        self.steps = None
        self.game_over = None
        
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize Game State
        self.cursor_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.score = 0
        self.time_remaining = self.MAX_TIME
        self.numbers = []
        self.particles = []
        self.combo_texts = []
        
        self.spawn_interval = self.INITIAL_SPAWN_INTERVAL * self.FPS
        self.spawn_timer = self.spawn_interval
        self.difficulty_timer = 0
        
        self.steps = 0
        self.game_over = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # --- Update Game Logic ---
        self._update_timers()
        self._handle_input(action)
        
        reward = self._update_game_entities()

        # --- Check Termination ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            # Optional: Add a "YOU WIN" message effect
        elif self.time_remaining <= 0:
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        
        # Truncated is always False in this environment
        truncated = False
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_timers(self):
        self.time_remaining = max(0, self.time_remaining - 1 / self.FPS)
        
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self._spawn_number()
            self.spawn_timer = self.spawn_interval
            
        self.difficulty_timer += 1
        if self.difficulty_timer >= self.DIFFICULTY_INTERVAL * self.FPS:
            new_interval_secs = max(self.MIN_SPAWN_INTERVAL, (self.spawn_interval / self.FPS) - self.SPAWN_RATE_INCREASE)
            self.spawn_interval = new_interval_secs * self.FPS
            self.difficulty_timer = 0
            
    def _handle_input(self, action):
        movement, _, _ = action
        
        if movement == 1:  # Up
            self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2:  # Down
            self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3:  # Left
            self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4:  # Right
            self.cursor_pos.x += self.CURSOR_SPEED
            
        # Clamp cursor position to screen bounds
        self.cursor_pos.x = np.clip(self.cursor_pos.x, self.CURSOR_SIZE / 2, self.WIDTH - self.CURSOR_SIZE / 2)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, self.CURSOR_SIZE / 2, self.HEIGHT - self.CURSOR_SIZE / 2)

    def _spawn_number(self):
        self.numbers.append({
            "pos": pygame.Vector2(self.np_random.uniform(20, self.WIDTH - 20), -20),
            "value": self.np_random.integers(1, 10),
            "vel": self.NUMBER_START_VEL,
            "alpha": 0,
            "size": 18 + self.np_random.integers(0, 5) # Slight size variation
        })
        
    def _update_game_entities(self):
        step_reward = 0
        
        # Update numbers and check for captures
        captured_this_frame = []
        cursor_rect = pygame.Rect(
            self.cursor_pos.x - self.CURSOR_SIZE / 2,
            self.cursor_pos.y - self.CURSOR_SIZE / 2,
            self.CURSOR_SIZE, self.CURSOR_SIZE
        )
        
        for number in self.numbers[:]:
            # Update position and fade-in
            number["pos"].y += number["vel"]
            number["vel"] += self.NUMBER_ACCELERATION
            number["alpha"] = min(255, number["alpha"] + 25)
            
            # Check for capture
            num_rect = pygame.Rect(number["pos"].x - number["size"]/2, number["pos"].y - number["size"]/2, number["size"], number["size"])
            if cursor_rect.colliderect(num_rect):
                captured_this_frame.append(number)
                self.numbers.remove(number)
                # Sound effect placeholder: # pygame.mixer.Sound('capture.wav').play()

            # Remove if off-screen
            elif number["pos"].y > self.HEIGHT + 20:
                self.numbers.remove(number)

        # Process captures and chain reactions
        if captured_this_frame:
            chain_length = len(captured_this_frame)
            if chain_length >= 2: # Chain reaction for 2 or more
                total_value = sum(n['value'] for n in captured_this_frame)
                combo_score = total_value * chain_length
                self.score += combo_score
                
                # Reward: +1 for each number, +5*chain_length for the combo event
                step_reward += chain_length + (5 * chain_length)
                
                # Visual Feedback
                self._create_combo_text(f"x{chain_length} COMBO!", self.cursor_pos)
                self._create_particle_burst(self.cursor_pos, 30, chain_length)
                # Sound effect placeholder: # pygame.mixer.Sound('combo.wav').play()
            else: # Single capture
                self.score += captured_this_frame[0]['value']
                step_reward += 1 # +1 for a single capture
                self._create_particle_burst(self.cursor_pos, 5, 1)

        # Update particles
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

        # Update combo texts
        for ct in self.combo_texts[:]:
            ct["pos"].y -= 0.5
            ct["lifespan"] -= 1
            if ct["lifespan"] <= 0:
                self.combo_texts.remove(ct)
                
        return step_reward

    def _create_particle_burst(self, pos, count, power):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3 + power)
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                "lifespan": self.np_random.integers(15, 31),
                "size": self.np_random.uniform(1, 4)
            })
            
    def _create_combo_text(self, text, pos):
        self.combo_texts.append({
            "text": text,
            "pos": pos.copy(),
            "lifespan": 60 # 2 seconds at 30fps
        })

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

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"].x), int(p["pos"].y), int(p["size"]),
                (*self.COLOR_PARTICLE, alpha)
            )

        # Render numbers
        num_font_cache = {}
        for number in self.numbers:
            size = number["size"]
            if size not in num_font_cache:
                num_font_cache[size] = pygame.font.SysFont("monospace", int(size), bold=True)
            
            font = num_font_cache[size]
            text_surf = font.render(str(number["value"]), True, self.COLOR_NUMBER)
            text_surf.set_alpha(number["alpha"])
            text_rect = text_surf.get_rect(center=(int(number["pos"].x), int(number["pos"].y)))
            self.screen.blit(text_surf, text_rect)

        # Render cursor with glow
        cx, cy = int(self.cursor_pos.x), int(self.cursor_pos.y)
        cursor_rect = pygame.Rect(cx - self.CURSOR_SIZE / 2, cy - self.CURSOR_SIZE / 2, self.CURSOR_SIZE, self.CURSOR_SIZE)
        
        # Glow effect
        glow_size = self.CURSOR_SIZE * 2.5
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_CURSOR_GLOW, (glow_size/2, glow_size/2), glow_size/2)
        self.screen.blit(glow_surf, (cx - glow_size/2, cy - glow_size/2))
        
        # Main cursor
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, border_radius=3)
        
        # Render combo texts
        for ct in self.combo_texts:
            alpha = int(255 * min(1, ct["lifespan"] / 30))
            text_surf = self.font_combo.render(ct["text"], True, self.COLOR_COMBO)
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=(int(ct["pos"].x), int(ct["pos"].y)))
            self.screen.blit(text_surf, text_rect)
            
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        time_text = self.font_ui.render(f"TIME: {self.time_remaining:.1f}", True, self.COLOR_UI)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
        }
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run in a headless environment
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11"
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Number Rain")
        clock = pygame.time.Clock()
        
        terminated = False
        total_reward = 0
        
        while not terminated:
            movement_action = 0 # No-op
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                movement_action = 1
            elif keys[pygame.K_DOWN]:
                movement_action = 2
            elif keys[pygame.K_LEFT]:
                movement_action = 3
            elif keys[pygame.K_RIGHT]:
                movement_action = 4
            
            action = [movement_action, 0, 0] # Space and Shift are not used for manual play
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Render the observation from the environment to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(GameEnv.FPS)

        print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
        env.close()
    except pygame.error as e:
        print(f"Could not run in graphical mode: {e}")
        print("This is expected in a headless environment. The environment itself is likely working correctly.")