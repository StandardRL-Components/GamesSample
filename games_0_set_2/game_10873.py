import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a neon-themed projectile arcade game.

    The player controls a launcher at the bottom of the screen. The goal is to
    shoot projectiles at targets to score points. The core mechanic involves
    toggling the projectile type between a straight shot and an arcing shot
    to hit targets and create chain-reaction combos.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `action[0]`: Movement (0=none, 1=up for angle--, 2=down for angle++)
    - `action[1]`: Space button (0=released, 1=held) -> Fires projectile on press
    - `action[2]`: Shift button (0=released, 1=held) -> Toggles projectile type on press

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Rewards:**
    - Hit target: + (0.5 * combo_multiplier), max 8.0
    - New max combo: +5
    - Miss: -2
    - Win (score >= 500 & accuracy >= 90%): +100
    - Lose (score < 0): -100

    **Termination:**
    - Score >= 500 and accuracy >= 90%
    - Score < 0
    - Episode steps >= 1000
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Shoot projectiles at targets in this neon-themed arcade game. Toggle between straight and "
        "arcing shots to hit targets and create chain-reaction combos."
    )
    user_guide = (
        "Use ↑ and ↓ arrow keys to aim the launcher. Press space to shoot and shift to toggle "
        "between straight and arcing projectiles."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1000

    # --- Colors (Neon on Dark) ---
    COLOR_BG = (10, 10, 20)
    COLOR_WHITE = (240, 240, 240)
    COLOR_TEXT = (200, 200, 220)
    COLOR_TARGET = (255, 255, 255)
    COLOR_PROJECTILE_STRAIGHT = (0, 255, 255) # Cyan
    COLOR_PROJECTILE_ARC = (255, 0, 255)     # Magenta
    
    # Particle colors by combo
    PARTICLE_COLORS = [
        (0, 255, 100),   # Green (1x)
        (255, 255, 0),   # Yellow (2x)
        (255, 165, 0),   # Orange (4x)
        (255, 50, 50),   # Red (8x+)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.hits = 0
        self.misses = 0
        self.combo = 0
        self.max_combo = 0
        
        self.launcher_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 20)
        self.launcher_angle = 90.0  # Degrees, 90 is straight up
        self.angle_speed = 2.0
        
        self.projectile = None
        self.projectile_type = "straight" # "straight" or "arc"
        self.launch_speed = 15.0
        self.gravity = 0.5

        self.targets = []
        self.num_targets = 5
        
        self.particles = []

        # --- Input State ---
        self.prev_space_held = False
        self.prev_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.hits = 0
        self.misses = 0
        self.combo = 0
        self.max_combo = 0
        
        self.launcher_angle = 90.0
        self.projectile_type = "straight"
        self.projectile = None
        
        self.particles = []
        self.targets = []
        self._spawn_targets(self.num_targets)

        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Input ---
        # Angle adjustment
        if movement == 1: # Up
            self.launcher_angle = min(170.0, self.launcher_angle + self.angle_speed)
        elif movement == 2: # Down
            self.launcher_angle = max(10.0, self.launcher_angle - self.angle_speed)
            
        # Toggle projectile type on key press (rising edge)
        if shift_held and not self.prev_shift_held:
            self.projectile_type = "arc" if self.projectile_type == "straight" else "straight"
            
        # Launch projectile on key press (rising edge)
        if space_held and not self.prev_space_held and self.projectile is None:
            self._launch_projectile()
            
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Update Game Logic ---
        hit_this_step = self._update_projectile()
        self._update_particles()
        
        # --- Calculate Reward ---
        if hit_this_step:
            self.hits += 1
            self.combo += 1
            
            combo_multiplier = min(16, 2**(self.combo - 1))
            score_gain = int(10 * combo_multiplier)
            self.score += score_gain
            
            reward += 0.5 * combo_multiplier # Scaled reward for RL
            
            if self.combo > self.max_combo:
                self.max_combo = self.combo
                reward += 5.0 # Bonus for new max combo
                
        elif hit_this_step is False: # Explicit miss
            self.misses += 1
            self.combo = 0
            self.score -= 20
            reward -= 2.0
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.score >= 500 and self._get_accuracy() >= 0.9:
                reward += 100 # Win bonus
            elif self.score < 0:
                reward -= 100 # Lose penalty
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "accuracy": self._get_accuracy(),
            "combo": self.combo,
        }

    def _get_accuracy(self):
        total_shots = self.hits + self.misses
        return self.hits / total_shots if total_shots > 0 else 1.0

    def _launch_projectile(self):
        angle_rad = math.radians(self.launcher_angle)
        vel_x = self.launch_speed * math.cos(angle_rad)
        vel_y = -self.launch_speed * math.sin(angle_rad) # Pygame y is inverted
        
        self.projectile = {
            "pos": list(self.launcher_pos),
            "vel": [vel_x, vel_y],
            "type": self.projectile_type,
            "radius": 8,
        }

    def _update_projectile(self):
        if self.projectile is None:
            return None # No event

        # Update position
        self.projectile["pos"][0] += self.projectile["vel"][0]
        self.projectile["pos"][1] += self.projectile["vel"][1]

        # Apply gravity for arc type
        if self.projectile["type"] == "arc":
            self.projectile["vel"][1] += self.gravity

        # Check for collision with targets
        for i, target in enumerate(self.targets):
            dist = math.hypot(
                self.projectile["pos"][0] - target["pos"][0],
                self.projectile["pos"][1] - target["pos"][1]
            )
            if dist < self.projectile["radius"] + target["radius"]:
                self._create_particles(target["pos"], self.combo)
                self._respawn_target(i)
                self.projectile = None
                return True # Hit event

        # Check for off-screen
        px, py = self.projectile["pos"]
        if not (0 < px < self.SCREEN_WIDTH and 0 < py < self.SCREEN_HEIGHT):
            self.projectile = None
            return False # Miss event
        
        return None # No event

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            p["radius"] = max(0, p["radius"] - 0.1)

    def _spawn_targets(self, count):
        for _ in range(count):
            while True:
                radius = self.np_random.integers(15, 31)
                pos = [
                    self.np_random.integers(radius, self.SCREEN_WIDTH - radius),
                    self.np_random.integers(radius, self.SCREEN_HEIGHT - 100),
                ]
                # Ensure no overlap with existing targets
                if not any(math.hypot(pos[0] - t["pos"][0], pos[1] - t["pos"][1]) < radius + t["radius"] + 10 for t in self.targets):
                    self.targets.append({"pos": pos, "radius": radius})
                    break

    def _respawn_target(self, index):
        self.targets.pop(index)
        self._spawn_targets(1)

    def _check_termination(self):
        accuracy = self._get_accuracy()
        if self.score >= 500 and accuracy >= 0.9:
            return True # Win
        if self.score < 0:
            return True # Lose
        return False

    def _render_glow_circle(self, surface, color, center, radius, max_alpha):
        if radius <= 0: return
        center_i = (int(center[0]), int(center[1]))
        for i in range(int(radius), 0, -2):
            alpha = max_alpha * (i / radius)
            temp_surf = pygame.Surface((int(radius * 2), int(radius * 2)), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*color, alpha), (int(radius), int(radius)), i)
            surface.blit(temp_surf, (center_i[0] - int(radius), center_i[1] - int(radius)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_game(self):
        # Draw targets
        for target in self.targets:
            pos_i = (int(target["pos"][0]), int(target["pos"][1]))
            pygame.gfxdraw.aacircle(self.screen, pos_i[0], pos_i[1], int(target["radius"]), self.COLOR_TARGET)
            pygame.gfxdraw.aacircle(self.screen, pos_i[0], pos_i[1], int(target["radius"]-1), self.COLOR_TARGET)

        # Draw launcher
        angle_rad = math.radians(self.launcher_angle)
        end_pos = (
            self.launcher_pos[0] + 40 * math.cos(angle_rad),
            self.launcher_pos[1] - 40 * math.sin(angle_rad)
        )
        pygame.draw.line(self.screen, self.COLOR_WHITE, self.launcher_pos, end_pos, 3)
        pygame.draw.circle(self.screen, self.COLOR_WHITE, self.launcher_pos, 10)

        # Draw projectile
        if self.projectile:
            proj_color = self.COLOR_PROJECTILE_STRAIGHT if self.projectile["type"] == "straight" else self.COLOR_PROJECTILE_ARC
            self._render_glow_circle(self.screen, proj_color, self.projectile["pos"], self.projectile["radius"] * 2, 100)
            pygame.draw.circle(self.screen, self.COLOR_WHITE, [int(p) for p in self.projectile["pos"]], int(self.projectile["radius"]))

        # Draw particles
        for p in self.particles:
            pos_i = (int(p["pos"][0]), int(p["pos"][1]))
            alpha = int(255 * (p["lifespan"] / p["initial_lifespan"]))
            pygame.gfxdraw.filled_circle(self.screen, pos_i[0], pos_i[1], int(p["radius"]), (*p["color"], alpha))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Accuracy
        acc_percent = self._get_accuracy() * 100
        acc_text = self.font_main.render(f"ACC: {acc_percent:.1f}%", True, self.COLOR_TEXT)
        self.screen.blit(acc_text, (self.SCREEN_WIDTH - acc_text.get_width() - 10, 10))
        
        # Combo
        if self.combo > 1:
            combo_text = self.font_main.render(f"{self.combo}x COMBO!", True, self.PARTICLE_COLORS[min(len(self.PARTICLE_COLORS)-1, self.combo-1)])
            self.screen.blit(combo_text, (self.SCREEN_WIDTH // 2 - combo_text.get_width() // 2, 50))

        # Projectile Type Indicator
        type_text_str = f"MODE: {self.projectile_type.upper()}"
        type_color = self.COLOR_PROJECTILE_STRAIGHT if self.projectile_type == "straight" else self.COLOR_PROJECTILE_ARC
        type_text = self.font_small.render(type_text_str, True, type_color)
        
        indicator_rect = pygame.Rect(0, 0, type_text.get_width() + 20, type_text.get_height() + 10)
        indicator_rect.center = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 50)
        
        pygame.draw.rect(self.screen, self.COLOR_BG, indicator_rect)
        pygame.draw.rect(self.screen, type_color, indicator_rect, 1)
        self.screen.blit(type_text, (indicator_rect.x + 10, indicator_rect.y + 5))

    def _create_particles(self, pos, combo):
        num_particles = 20 + min(combo * 5, 50)
        color_index = min(len(self.PARTICLE_COLORS) - 1, max(0, combo - 1))
        particle_color = self.PARTICLE_COLORS[color_index]
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "lifespan": lifespan,
                "initial_lifespan": lifespan,
                "radius": self.np_random.uniform(2, 5),
                "color": particle_color,
            })
    
    def close(self):
        pygame.quit()


# --- Example Usage ---
if __name__ == "__main__":
    # To run with display, unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    
    # --- Manual Play ---
    pygame.display.set_caption("Projectile Chain Reaction")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("R: Reset")

    while not done:
        # Action mapping for human player
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Accuracy: {info['accuracy']:.2f}, Reward: {reward:.2f}")

        if terminated or truncated:
            print(f"--- Episode Finished ---")
            print(f"Final Score: {info['score']}, Final Accuracy: {info['accuracy']:.2f}")
            obs, info = env.reset()

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)
        
    env.close()