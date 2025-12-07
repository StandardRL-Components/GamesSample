import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:07:19.563735
# Source Brief: brief_02361.md
# Brief Index: 2361
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Cellular Conquest'.

    The player controls a cell in a microscopic world, absorbing smaller cells
    to grow and firing projectiles to shrink larger rivals.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `actions[0]` (Movement): 0=None, 1=Up, 2=Down, 3=Left, 4=Right
    - `actions[1]` (Fire): 0=Released, 1=Held (fires projectile)
    - `actions[2]` (Unused): 0=Released, 1=Held

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Rewards:**
    - +0.1 for absorbing a smaller cell.
    - -0.01 for colliding with a larger cell.
    - +1.0 for reaching a size milestone (50, 100).
    - +100 for winning (reaching max size).
    - -100 for losing (size shrinks to zero).

    **Termination:**
    - Player cell size <= 0.
    - Player cell size >= 200.
    - Episode length > 5000 steps.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Control a cell in a microscopic world. Absorb smaller cells to grow and fire projectiles to shrink larger rivals."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move your cell. Press space to fire a projectile in your last direction of movement."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000
        self.WIN_SIZE = 200
        self.MIN_SIZE = 1

        # Player
        self.PLAYER_INITIAL_SIZE = 15
        self.PLAYER_SPEED = 4.0
        self.PLAYER_FIRE_COOLDOWN = 15  # frames
        self.PLAYER_MILESTONE_1 = 50
        self.PLAYER_MILESTONE_2 = 100

        # Rivals
        self.INITIAL_RIVALS = 15
        self.MAX_RIVALS = 25
        self.RIVAL_SPAWN_RATE = 60 # frames
        self.RIVAL_MIN_SPEED = 0.5
        self.RIVAL_MAX_SPEED = 2.0
        self.RIVAL_PASSIVE_GROWTH = 0.005 # size per frame
        self.RIVAL_MAX_SIZE_INCREMENT_STEPS = 1000
        self.RIVAL_SPEED_INCREMENT_STEPS = 500
        
        # Projectiles
        self.PROJECTILE_SPEED = 8.0
        self.PROJECTILE_SIZE = 4
        self.PROJECTILE_SHRINK_DURATION = 90 # frames (3 seconds)
        self.PROJECTILE_SHRINK_RATE = 0.01 # % of current size per frame

        # Colors
        self.COLOR_BG_START = (5, 10, 20)
        self.COLOR_BG_END = (20, 5, 10)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 50)
        self.COLOR_RIVAL_SMALL = (255, 100, 100)
        self.COLOR_RIVAL_LARGE = (255, 0, 0)
        self.COLOR_PROJECTILE = (100, 200, 255)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_BAR_BG = (50, 50, 50)
        self.COLOR_BAR_FG = (0, 200, 100)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.player = {}
        self.rivals = []
        self.projectiles = []
        
        self.rival_max_spawn_size = 0
        self.current_rival_speed_multiplier = 0
        self.next_rival_spawn_timer = 0
        self.milestone1_reached = False
        self.milestone2_reached = False

        # self.reset() is called by the environment wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False

        self.player = {
            "pos": pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2),
            "size": self.PLAYER_INITIAL_SIZE,
            "fire_cooldown": 0,
            "last_fire_direction": pygame.math.Vector2(0, -1),
            "projectile_range": 300,
            "fire_rate_multiplier": 1.0,
        }

        self.rivals = []
        self.projectiles = []
        
        self.rival_max_spawn_size = 20
        self.current_rival_speed_multiplier = 1.0
        self.next_rival_spawn_timer = self.RIVAL_SPAWN_RATE

        self.milestone1_reached = False
        self.milestone2_reached = False

        for _ in range(self.INITIAL_RIVALS):
            self._spawn_rival(is_initial=True)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0.0

        self._handle_input(action)
        
        reward += self._update_player()
        self._update_projectiles()
        self._update_rivals()
        reward += self._handle_collisions()
        self._spawn_new_rivals()

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated:
            if self.player["size"] >= self.WIN_SIZE:
                reward = 100.0
            else:
                reward = -100.0
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # --- Movement ---
        direction = pygame.math.Vector2(0, 0)
        if movement == 1: direction.y = -1 # Up
        elif movement == 2: direction.y = 1 # Down
        elif movement == 3: direction.x = -1 # Left
        elif movement == 4: direction.x = 1 # Right

        if direction.length() > 0:
            direction.normalize_ip()
            self.player["last_fire_direction"] = direction.copy()
        
        self.player["pos"] += direction * self.PLAYER_SPEED

        # --- Firing ---
        if space_held and self.player["fire_cooldown"] <= 0:
            # SFX: Player shoot
            self.player["fire_cooldown"] = self.PLAYER_FIRE_COOLDOWN / self.player["fire_rate_multiplier"]
            
            start_pos = self.player["pos"] + self.player["last_fire_direction"] * self.player["size"]
            projectile = {
                "pos": start_pos,
                "vel": self.player["last_fire_direction"] * self.PROJECTILE_SPEED,
                "lifetime": self.player["projectile_range"] / self.PROJECTILE_SPEED
            }
            self.projectiles.append(projectile)

    def _update_player(self):
        self._wrap_position(self.player["pos"])
        if self.player["fire_cooldown"] > 0:
            self.player["fire_cooldown"] -= 1

        # Check for upgrades and award milestone reward
        reward = 0
        if not self.milestone1_reached and self.player["size"] >= self.PLAYER_MILESTONE_1:
            self.milestone1_reached = True
            self.player["projectile_range"] = 500 # Increase range
            reward += 1.0 # Milestone reward
        if not self.milestone2_reached and self.player["size"] >= self.PLAYER_MILESTONE_2:
            self.milestone2_reached = True
            self.player["fire_rate_multiplier"] = 1.5 # Increase fire rate
            reward += 1.0 # Milestone reward
        return reward

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p["pos"] += p["vel"]
            p["lifetime"] -= 1
            
            # Check for collision with rivals
            hit = False
            for r in self.rivals:
                dist = p["pos"].distance_to(r["pos"])
                if dist < r["size"] + self.PROJECTILE_SIZE:
                    # SFX: Projectile hit
                    r["debuffs"].append({
                        "timer": self.PROJECTILE_SHRINK_DURATION,
                        "rate": self.PROJECTILE_SHRINK_RATE
                    })
                    hit = True
                    break # Projectile hits one target and disappears
            
            if hit or p["lifetime"] <= 0 or not (0 < p["pos"].x < self.WIDTH and 0 < p["pos"].y < self.HEIGHT):
                self.projectiles.remove(p)

    def _update_rivals(self):
        # Update difficulty scaling
        if self.steps > 0 and self.steps % self.RIVAL_MAX_SIZE_INCREMENT_STEPS == 0:
            self.rival_max_spawn_size += 5
        if self.steps > 0 and self.steps % self.RIVAL_SPEED_INCREMENT_STEPS == 0:
            self.current_rival_speed_multiplier += 0.05

        for r in self.rivals:
            # Random movement
            if self.np_random.random() < 0.02: # Change direction occasionally
                r["vel"].rotate_ip(self.np_random.uniform(-45, 45))
                r["vel"].scale_to_length(self.np_random.uniform(self.RIVAL_MIN_SPEED, self.RIVAL_MAX_SPEED) * self.current_rival_speed_multiplier)

            r["pos"] += r["vel"]
            self._wrap_position(r["pos"])
            
            # Apply debuffs
            total_shrink = 0
            for debuff in r["debuffs"][:]:
                total_shrink += r["size"] * debuff["rate"]
                debuff["timer"] -= 1
                if debuff["timer"] <= 0:
                    r["debuffs"].remove(debuff)
            
            r["size"] = max(self.MIN_SIZE, r["size"] - total_shrink + self.RIVAL_PASSIVE_GROWTH)
            
    def _handle_collisions(self):
        reward = 0
        for r in self.rivals[:]:
            dist = self.player["pos"].distance_to(r["pos"])
            if dist < self.player["size"] + r["size"]:
                # Using a 1.1x size advantage rule to absorb
                if self.player["size"] > r["size"] * 1.1:
                    # SFX: Absorb cell
                    # Absorb: gain area, not radius
                    player_area = math.pi * self.player["size"]**2
                    rival_area = math.pi * r["size"]**2
                    self.player["size"] = math.sqrt((player_area + rival_area) / math.pi)
                    self.rivals.remove(r)
                    reward += 0.1
                else:
                    # SFX: Player takes damage
                    # Collision with larger or similar size cell
                    self.player["size"] = max(self.MIN_SIZE, self.player["size"] - 0.2)
                    reward -= 0.01
        return reward

    def _spawn_new_rivals(self):
        self.next_rival_spawn_timer -= 1
        if self.next_rival_spawn_timer <= 0 and len(self.rivals) < self.MAX_RIVALS:
            self._spawn_rival()
            self.next_rival_spawn_timer = self.RIVAL_SPAWN_RATE

        # Anti-softlock: ensure there's always small food
        small_cells = sum(1 for r in self.rivals if r["size"] < self.player["size"])
        if small_cells < 3:
            self._spawn_rival(force_small=True)

    def _spawn_rival(self, is_initial=False, force_small=False):
        if force_small:
            size = self.np_random.uniform(5, max(6, self.player["size"] * 0.5))
        else:
            size = self.np_random.uniform(5, self.rival_max_spawn_size)
        
        # Spawn away from the player
        edge = self.np_random.integers(4)
        if edge == 0: pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), -size)
        elif edge == 1: pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + size)
        elif edge == 2: pos = pygame.math.Vector2(-size, self.np_random.uniform(0, self.HEIGHT))
        else: pos = pygame.math.Vector2(self.WIDTH + size, self.np_random.uniform(0, self.HEIGHT))
        
        if is_initial: # Scatter initial cells randomly
            pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT))
        
        vel = pygame.math.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
        if vel.length() > 0:
            vel.scale_to_length(self.np_random.uniform(self.RIVAL_MIN_SPEED, self.RIVAL_MAX_SPEED))
        
        self.rivals.append({"pos": pos, "size": size, "vel": vel, "debuffs": []})

    def _check_termination(self):
        if self.player["size"] < self.MIN_SIZE:
            self.game_over = True
            return True
        if self.player["size"] >= self.WIN_SIZE:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _wrap_position(self, pos):
        pos.x = pos.x % self.WIDTH
        pos.y = pos.y % self.HEIGHT

    def _get_observation(self):
        self._render_background()
        self._render_rivals()
        self._render_projectiles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_size": self.player["size"],
            "rivals_count": len(self.rivals)
        }

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_START[0] * (1 - interp) + self.COLOR_BG_END[0] * interp),
                int(self.COLOR_BG_START[1] * (1 - interp) + self.COLOR_BG_END[1] * interp),
                int(self.COLOR_BG_START[2] * (1 - interp) + self.COLOR_BG_END[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_rivals(self):
        for r in sorted(self.rivals, key=lambda x: x["size"]): # Draw smaller cells first
            pos_x, pos_y = int(r["pos"].x), int(r["pos"].y)
            size = int(r["size"])
            if size <= 0: continue

            # Interpolate color based on size
            max_size_for_color = 100
            interp = min(1.0, r["size"] / max_size_for_color)
            color = (
                int(self.COLOR_RIVAL_SMALL[0] * (1 - interp) + self.COLOR_RIVAL_LARGE[0] * interp),
                int(self.COLOR_RIVAL_SMALL[1] * (1 - interp) + self.COLOR_RIVAL_LARGE[1] * interp),
                int(self.COLOR_RIVAL_SMALL[2] * (1 - interp) + self.COLOR_RIVAL_LARGE[2] * interp)
            )

            # Pulsating effect
            pulse = math.sin(self.steps * 0.1 + r["pos"].x) * (size * 0.05)
            render_size = int(max(1, size + pulse))

            # Debuff visual effect (blueish tint)
            if r["debuffs"]:
                color = tuple(int(c * 0.7 + self.COLOR_PROJECTILE[i] * 0.3) for i, c in enumerate(color))

            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, render_size, color)
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, render_size, color)

    def _render_projectiles(self):
        for p in self.projectiles:
            pos_x, pos_y = int(p["pos"].x), int(p["pos"].y)
            # Fade out effect
            alpha = max(0, min(255, int(255 * (p["lifetime"] / 15))))
            color = (*self.COLOR_PROJECTILE, alpha)
            
            s = pygame.Surface((self.PROJECTILE_SIZE*2, self.PROJECTILE_SIZE*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (self.PROJECTILE_SIZE, self.PROJECTILE_SIZE), self.PROJECTILE_SIZE)
            self.screen.blit(s, (pos_x - self.PROJECTILE_SIZE, pos_y - self.PROJECTILE_SIZE))

    def _render_player(self):
        pos_x, pos_y = int(self.player["pos"].x), int(self.player["pos"].y)
        size = int(self.player["size"])
        if size <= 0: return

        # Pulsating effect
        pulse = math.sin(self.steps * 0.15) * (size * 0.05)
        render_size = int(max(1, size + pulse))
        
        # Glow effect
        glow_size = int(render_size * 1.5)
        s = pygame.Surface((glow_size*2, glow_size*2), pygame.SRCALPHA)
        pygame.draw.circle(s, self.COLOR_PLAYER_GLOW, (glow_size, glow_size), glow_size)
        self.screen.blit(s, (pos_x - glow_size, pos_y - glow_size))

        # Main body
        pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, render_size, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, render_size, self.COLOR_PLAYER)

    def _render_ui(self):
        # Size display
        size_text = self.font_main.render(f"Size: {self.player['size']:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(size_text, (10, 10))

        # Upgrade progress bar
        if self.player["size"] < self.PLAYER_MILESTONE_1:
            prev_m, next_m, text = 0, self.PLAYER_MILESTONE_1, "-> Range"
        elif self.player["size"] < self.PLAYER_MILESTONE_2:
            prev_m, next_m, text = self.PLAYER_MILESTONE_1, self.PLAYER_MILESTONE_2, "-> Fire Rate"
        else:
            prev_m, next_m, text = self.PLAYER_MILESTONE_2, self.WIN_SIZE, "-> WIN"
        
        progress = (self.player["size"] - prev_m) / max(1, (next_m - prev_m))
        progress = max(0, min(1, progress))

        bar_width = 150
        bar_height = 12
        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (10, 40, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_BAR_FG, (10, 40, int(bar_width * progress), bar_height))
        
        progress_text = self.font_small.render(text, True, self.COLOR_TEXT)
        self.screen.blit(progress_text, (20 + bar_width, 38))

        # Game Over text
        if self.game_over:
            outcome_text = "VICTORY!" if self.player["size"] >= self.WIN_SIZE else "DEFEATED"
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            text_surf = self.font_main.render(outcome_text, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # This method is for self-testing and can be removed in production
        print("✓ Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs, _ = self.reset()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows for human playtesting
    # To play, you need to have pygame installed and not run in a headless environment
    # Comment out or change the SDL_VIDEODRIVER line at the top of the file
    # e.g., os.environ.pop("SDL_VIDEODRIVER", None)
    
    # For this script to run as-is, we will keep it headless and just run a few steps
    print("Running a short headless test...")
    env = GameEnv()
    try:
        env.validate_implementation()
        obs, info = env.reset()
        for i in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Episode finished at step {i+1}.")
                break
        print("Headless test completed successfully.")
    except Exception as e:
        print(f"An error occurred during the test: {e}")
    finally:
        env.close()