import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T19:13:45.367347
# Source Brief: brief_03143.md
# Brief Index: 3143
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium Environment: Pin Knockdown
    -----------------------------------
    The player controls a launcher at the bottom of the screen. The goal is to
    launch a projectile to knock down moving pins of different values. Hitting
    pins increases a momentum bonus, making subsequent shots faster. Missing a
    shot resets the momentum and costs a life.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
        - Left/Right adjusts the launch angle. Up/Down have no effect.
    - actions[1]: Space button (0=released, 1=held)
        - Launches the projectile if it's ready.
    - actions[2]: Shift button (0=released, 1=held)
        - No effect.

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - Hit a pin: +0.1 base + pin-specific value (+1, +2.5, +5).
    - Miss a shot (projectile goes off-screen): -0.1 base - 10 for life loss.
    - Win the game (score >= 500): +100.
    - Lose the game (lives <= 0): -50.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Launch projectiles to knock down moving pins and score points. "
        "Hit consecutive pins to build momentum for faster shots, but a miss will cost a life."
    )
    user_guide = "Controls: Use ←→ arrow keys to aim the launcher. Press space to fire a projectile."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    WIN_SCORE = 500
    INITIAL_LIVES = 3
    NUM_PINS = 7
    GRAVITY = 0.15
    PROJECTILE_BASE_SPEED = 7.0
    MOMENTUM_INCREASE_PER_HIT = 0.2

    # --- Colors ---
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (40, 45, 50)
    COLOR_PLAYER = (255, 200, 0)
    COLOR_PROJECTILE = (255, 255, 0)
    COLOR_PROJECTILE_GLOW = (255, 255, 0, 64)
    COLOR_TEXT = (220, 220, 220)
    COLOR_PIN_GREEN = (0, 255, 127)
    COLOR_PIN_BLUE = (0, 191, 255)
    COLOR_PIN_RED = (255, 69, 0)
    PIN_DATA = {
        "green": {"color": COLOR_PIN_GREEN, "value": 10, "reward": 1.0},
        "blue": {"color": COLOR_PIN_BLUE, "value": 25, "reward": 2.5},
        "red": {"color": COLOR_PIN_RED, "value": 50, "reward": 5.0},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.launch_angle = 0.0
        self.momentum_bonus = 0.0
        self.projectile = None
        self.pins = []
        self.particles = []

        # self.reset() # reset is called by the wrapper
        
        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        
        self.launch_angle = 0.0 # Straight up
        self.momentum_bonus = 0.0

        self.projectile = {
            "active": False,
            "pos": pygame.Vector2(0, 0),
            "vel": pygame.Vector2(0, 0),
            "radius": 8
        }
        self.pins = []
        self.particles = []
        
        self._ensure_pin_count()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1

        movement, space_pressed, _ = action
        
        # --- 1. Handle Player Input ---
        self._handle_input(movement, space_pressed)

        # --- 2. Update Game Logic ---
        # Update projectile
        if self.projectile["active"]:
            hit_reward, miss_penalty = self._update_projectile()
            reward += hit_reward
            reward += miss_penalty

        # Update pins
        self._update_pins()
        self._ensure_pin_count()
        
        # Update particles
        self._update_particles()

        # --- 3. Calculate Reward & Check Termination ---
        terminated = False
        
        if self.score >= self.WIN_SCORE and not self.game_over:
            reward += 100
            self.game_over = True
            terminated = True
        elif self.lives <= 0 and not self.game_over:
            reward -= 50
            self.game_over = True
            terminated = True
        
        truncated = self.steps >= self.MAX_STEPS
        terminated = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed):
        # Adjust launch angle
        if movement == 3: # Left
            self.launch_angle = max(-80.0, self.launch_angle - 2.0)
        elif movement == 4: # Right
            self.launch_angle = min(80.0, self.launch_angle + 2.0)

        # Launch projectile
        if space_pressed and not self.projectile["active"]:
            # SFX: Launch sound
            self.projectile["active"] = True
            self.projectile["pos"] = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 40)
            
            speed = self.PROJECTILE_BASE_SPEED + self.momentum_bonus
            angle_rad = math.radians(self.launch_angle - 90) # Adjust for Pygame coords

            self.projectile["vel"] = pygame.Vector2(
                math.cos(angle_rad) * speed,
                math.sin(angle_rad) * speed
            )

    def _update_projectile(self):
        hit_reward = 0
        miss_penalty = 0

        # Apply gravity
        self.projectile["vel"].y += self.GRAVITY
        # Update position
        self.projectile["pos"] += self.projectile["vel"]

        # Check for collision with pins
        proj_pos = self.projectile["pos"]
        for pin in self.pins[:]:
            if pin["rect"].collidepoint(proj_pos.x, proj_pos.y):
                # SFX: Pin hit sound
                self.score += pin["value"]
                hit_reward += 0.1 + pin["reward"]
                self.momentum_bonus += self.MOMENTUM_INCREASE_PER_HIT
                self._create_particles(pin["rect"].center, pin["color"], 20)
                self.pins.remove(pin)
                self.projectile["active"] = False # Projectile disappears on hit
                break
        
        # Check for out of bounds (miss)
        if self.projectile["active"]:
            if not (0 < proj_pos.x < self.SCREEN_WIDTH and proj_pos.y < self.SCREEN_HEIGHT):
                # SFX: Miss sound
                self.lives -= 1
                miss_penalty -= 10.1 # -10 for life loss, -0.1 for miss
                self.momentum_bonus = 0 # Reset momentum on miss
                self.projectile["active"] = False

        return hit_reward, miss_penalty

    def _update_pins(self):
        for pin in self.pins:
            pin["rect"].x = pin["path"]["center_x"] + math.sin(self.steps * pin["path"]["speed"] + pin["path"]["phase"]) * pin["path"]["amplitude"]

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _ensure_pin_count(self):
        while len(self.pins) < self.NUM_PINS:
            pin_type_name = random.choice(list(self.PIN_DATA.keys()))
            pin_data = self.PIN_DATA[pin_type_name]
            
            pin_rect = pygame.Rect(0, 0, 20, 40)
            pin_rect.centery = random.randint(50, self.SCREEN_HEIGHT // 2)
            
            path_params = {
                "center_x": random.randint(100, self.SCREEN_WIDTH - 100),
                "amplitude": random.randint(50, 150),
                "speed": random.uniform(0.01, 0.03),
                "phase": random.uniform(0, 2 * math.pi)
            }
            
            new_pin = {
                "rect": pin_rect,
                "color": pin_data["color"],
                "value": pin_data["value"],
                "reward": pin_data["reward"],
                "path": path_params
            }
            self.pins.append(new_pin)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lives": self.lives, "momentum": self.momentum_bonus}
    
    def _render_game(self):
        # Draw background grid
        for i in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

        # Draw pins
        for pin in self.pins:
            pygame.draw.rect(self.screen, pin["color"], pin["rect"], border_radius=4)
            darker_color = tuple(c * 0.7 for c in pin["color"])
            pygame.draw.rect(self.screen, darker_color, pin["rect"], width=2, border_radius=4)

        # Draw launcher
        launcher_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 20)
        pygame.draw.rect(self.screen, (100, 100, 110), (launcher_pos[0] - 30, launcher_pos[1] - 10, 60, 20), border_radius=5)
        
        # Draw aiming line
        if not self.projectile["active"]:
            angle_rad = math.radians(self.launch_angle - 90)
            end_pos_x = launcher_pos[0] + math.cos(angle_rad) * 40
            end_pos_y = launcher_pos[1] - 20 + math.sin(angle_rad) * 40
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (launcher_pos[0], launcher_pos[1] - 20), (end_pos_x, end_pos_y), 3)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
            p_color = p["color"] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), p_color)

        # Draw projectile
        if self.projectile["active"]:
            pos = self.projectile["pos"]
            radius = self.projectile["radius"]
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius * 1.8), self.COLOR_PROJECTILE_GLOW)
            # Main circle
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), radius, self.COLOR_PROJECTILE)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, self.COLOR_PROJECTILE)

    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Lives display
        lives_text = self.font_main.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - 150, 10))
        for i in range(self.lives):
            pygame.draw.circle(self.screen, self.COLOR_PIN_RED, (self.SCREEN_WIDTH - 60 + i * 25, 22), 8)
            pygame.draw.circle(self.screen, (150, 0, 0), (self.SCREEN_WIDTH - 60 + i * 25, 22), 8, 2)

        # Momentum display
        momentum_text = self.font_main.render(f"SPEED: x{1 + self.momentum_bonus / self.PROJECTILE_BASE_SPEED:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(momentum_text, (self.SCREEN_WIDTH // 2 - momentum_text.get_width() // 2, 10))

        # Game Over text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WON!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = self.COLOR_PIN_GREEN if self.score >= self.WIN_SCORE else self.COLOR_PIN_RED
            
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            lifespan = random.randint(15, 30)
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                "color": color,
                "radius": random.randint(2, 5),
                "lifespan": lifespan,
                "max_lifespan": lifespan,
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("--- Running Implementation Validation ---")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2], f"Expected [5, 2, 2], got {self.action_space.nvec.tolist()}"
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Expected {(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)}, got {test_obs.shape}"
        assert test_obs.dtype == np.uint8, f"Expected uint8, got {test_obs.dtype}"
        
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # For this to work, you need to comment out the line:
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    
    # Example of how to create and use the environment
    try:
        env = GameEnv()
        obs, info = env.reset()
        done = False
        
        # To play manually, you would need a display.
        # This example just runs a few random steps.
        print("Running 5 random steps...")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: Reward={reward}, Terminated={terminated}, Truncated={truncated}, Info={info}")
            if terminated or truncated:
                print("Episode finished. Resetting.")
                obs, info = env.reset()
        env.close()
        print("Done.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("If you are trying to run this manually, make sure to comment out 'os.environ.setdefault(\"SDL_VIDEODRIVER\", \"dummy\")' at the top of the file.")
        print("And ensure you have a display environment (e.g., not running in a plain terminal/SSH session).")