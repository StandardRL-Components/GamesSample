import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:08:32.487017
# Source Brief: brief_01540.md
# Brief Index: 1540
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for "Dream City Defense".

    In this game, the agent must defend a central city foundation from
    encroaching nightmares by strategically placing defensive blocks. The
    city's health shrinks as nightmares reach the center. The agent wins
    by surviving for a set duration (until dawn).

    Visuals are a key focus, with a dreamlike, neon-on-dark aesthetic,
    particle effects, and smooth animations.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your dream city's foundation from encroaching nightmares by placing defensive blocks. "
        "Survive until dawn to win."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to place a block and shift to cycle "
        "between available block types."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1000  # Dawn arrives

    # Colors
    COLOR_BG_START = (10, 0, 30)
    COLOR_BG_END = (40, 0, 60)
    COLOR_FOUNDATION = (0, 50, 100)
    COLOR_FOUNDATION_GLOW = (0, 100, 150)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_NIGHTMARE = (255, 20, 50)
    COLOR_NIGHTMARE_GLOW = (180, 0, 30)
    COLOR_TEXT = (220, 220, 255)
    COLOR_UI_ACCENT = (0, 200, 200)

    # Game Parameters
    CURSOR_SPEED = 20
    GRID_SIZE = 40
    PLACEMENT_COOLDOWN_FRAMES = 15
    MAX_CITY_HEALTH = 1000
    NIGHTMARE_SPAWN_RATE_INITIAL = 0.02
    NIGHTMARE_SPAWN_RATE_INCREASE = 0.0005
    NIGHTMARE_BASE_HEALTH_INITIAL = 50
    NIGHTMARE_HEALTH_INCREASE_INTERVAL = 200
    NIGHTMARE_HEALTH_INCREASE_AMOUNT = 25
    NIGHTMARE_SPEED = 1.5
    NIGHTMARE_DAMAGE = 50 # Damage to foundation
    BLOCK_DAMAGE_PER_STEP = 2 # Damage nightmare deals to block

    BLOCK_TYPES = {
        "standard": {
            "size": (40, 40),
            "health": 100,
            "color": (0, 255, 150),
            "glow": (0, 150, 100)
        },
        "reinforced": {
            "size": (80, 40),
            "health": 250,
            "color": (0, 200, 255),
            "glow": (0, 120, 180)
        },
        "tall": {
            "size": (40, 80),
            "health": 250,
            "color": (150, 100, 255),
            "glow": (100, 70, 180)
        }
    }


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)

        # Pre-render background for performance
        self._background_surface = self._create_gradient_background()

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.city_health = 0
        self.cursor_pos = [0, 0]
        self.placement_cooldown = 0
        self.prev_shift_held = False
        self.current_block_type_idx = 0
        self.unlocked_block_types = []
        self.blocks = []
        self.nightmares = []
        self.particles = []
        self.nightmare_spawn_rate = 0
        self.nightmare_base_health = 0

        # self.reset() is called by the environment wrapper
        # self.validate_implementation() # Optional self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.city_health = self.MAX_CITY_HEALTH
        self.cursor_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2]
        self.placement_cooldown = 0
        self.prev_shift_held = False
        self.current_block_type_idx = 0
        self.unlocked_block_types = ["standard"]
        self.blocks = []
        self.nightmares = []
        self.particles = []
        self.nightmare_spawn_rate = self.NIGHTMARE_SPAWN_RATE_INITIAL
        self.nightmare_base_health = self.NIGHTMARE_BASE_HEALTH_INITIAL

        self.foundation_rect = pygame.Rect(
            self.SCREEN_WIDTH // 2 - 50,
            self.SCREEN_HEIGHT // 2 - 50,
            100, 100
        )

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        self.placement_cooldown = max(0, self.placement_cooldown - 1)

        # --- 1. Handle Actions ---
        placement_reward = self._handle_actions(action)
        reward += placement_reward

        # --- 2. Update Game Logic ---
        self._update_unlocks()
        self._spawn_nightmares()
        
        nightmare_rewards = self._update_nightmares_and_blocks()
        reward += nightmare_rewards

        self._update_particles()
        
        # --- 3. Calculate Continuous Reward & Check Termination ---
        # Small reward for maintaining city health
        reward += (self.city_health / self.MAX_CITY_HEALTH) * 0.01

        terminated = self._check_termination()
        if terminated:
            if self.steps >= self.MAX_STEPS:
                reward += 100  # Victory bonus
            else:
                reward -= 100  # Defeat penalty

        # Clamp reward to specified range for non-terminal steps
        if not terminated:
            reward = np.clip(reward, -10, 10)

        self.score += reward

        truncated = self.steps >= self.MAX_STEPS
        terminated = self.city_health <= 0

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Movement
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED

        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

        # Cycle block type (on key press, not hold)
        if shift_held and not self.prev_shift_held:
            self.current_block_type_idx = (self.current_block_type_idx + 1) % len(self.unlocked_block_types)
            # sfx: UI_switch.wav
        self.prev_shift_held = shift_held

        # Place block
        if space_held and self.placement_cooldown == 0:
            block_name = self.unlocked_block_types[self.current_block_type_idx]
            block_info = self.BLOCK_TYPES[block_name]
            block_size = block_info["size"]
            
            # Snap to grid
            grid_x = round((self.cursor_pos[0] - block_size[0] / 2) / self.GRID_SIZE) * self.GRID_SIZE
            grid_y = round((self.cursor_pos[1] - block_size[1] / 2) / self.GRID_SIZE) * self.GRID_SIZE

            new_block_rect = pygame.Rect(grid_x, grid_y, block_size[0], block_size[1])

            # Check for overlaps and out of bounds
            can_place = True
            if not self.screen.get_rect().contains(new_block_rect):
                can_place = False
            for block in self.blocks:
                if new_block_rect.colliderect(block["rect"]):
                    can_place = False
                    break
            
            if can_place:
                self.blocks.append({
                    "rect": new_block_rect,
                    "type": block_name,
                    "health": block_info["health"],
                    "max_health": block_info["health"]
                })
                self.placement_cooldown = self.PLACEMENT_COOLDOWN_FRAMES
                self._create_particles(new_block_rect.center, 10, block_info["color"])
                reward -= 0.5 # Cost for placing a block
                # sfx: place_block.wav
        return reward

    def _update_unlocks(self):
        if self.steps == 250 and "reinforced" not in self.unlocked_block_types:
            self.unlocked_block_types.append("reinforced")
        if self.steps == 500 and "tall" not in self.unlocked_block_types:
            self.unlocked_block_types.append("tall")
        # No unlock at 750, 3 types is enough complexity.

    def _spawn_nightmares(self):
        # Increase difficulty over time
        self.nightmare_spawn_rate += self.NIGHTMARE_SPAWN_RATE_INCREASE
        if self.steps > 0 and self.steps % self.NIGHTMARE_HEALTH_INCREASE_INTERVAL == 0:
            self.nightmare_base_health += self.NIGHTMARE_HEALTH_INCREASE_AMOUNT

        if self.np_random.random() < self.nightmare_spawn_rate:
            side = self.np_random.integers(0, 4)
            if side == 0: # Top
                pos = [self.np_random.uniform(0, self.SCREEN_WIDTH), -20]
            elif side == 1: # Bottom
                pos = [self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20]
            elif side == 2: # Left
                pos = [-20, self.np_random.uniform(0, self.SCREEN_HEIGHT)]
            else: # Right
                pos = [self.SCREEN_WIDTH + 20, self.np_random.uniform(0, self.SCREEN_HEIGHT)]
            
            self.nightmares.append({
                "pos": np.array(pos, dtype=float),
                "radius": self.np_random.uniform(8, 15),
                "health": self.nightmare_base_health,
                "max_health": self.nightmare_base_health
            })

    def _update_nightmares_and_blocks(self):
        reward = 0
        
        # Update nightmares
        for nightmare in self.nightmares[:]:
            target_pos = np.array(self.foundation_rect.center, dtype=float)
            direction = target_pos - nightmare["pos"]
            dist = np.linalg.norm(direction)
            
            if dist > 0:
                direction /= dist
            
            nightmare["pos"] += direction * self.NIGHTMARE_SPEED
            nightmare_rect = pygame.Rect(
                nightmare["pos"][0] - nightmare["radius"],
                nightmare["pos"][1] - nightmare["radius"],
                nightmare["radius"] * 2,
                nightmare["radius"] * 2
            )

            # Collision with foundation
            if nightmare_rect.colliderect(self.foundation_rect):
                self.city_health -= self.NIGHTMARE_DAMAGE
                self._create_particles(nightmare["pos"], 30, self.COLOR_NIGHTMARE, 2)
                self.nightmares.remove(nightmare)
                reward -= 5.0 # Penalty for foundation hit
                # sfx: foundation_damage.wav
                continue

            # Collision with blocks
            collided_with_block = False
            for block in self.blocks[:]:
                if nightmare_rect.colliderect(block["rect"]):
                    collided_with_block = True
                    block["health"] -= self.BLOCK_DAMAGE_PER_STEP
                    nightmare["health"] -= self.BLOCK_DAMAGE_PER_STEP
                    
                    if self.steps % 5 == 0: # Occasional sparks
                        self._create_particles(nightmare_rect.center, 2, (255, 255, 0))

                    if block["health"] <= 0:
                        self._create_particles(block["rect"].center, 40, (200, 200, 200), 2.5)
                        self.blocks.remove(block)
                        # sfx: block_destroy.wav

                    if nightmare["health"] <= 0:
                        self._create_particles(nightmare_rect.center, 25, self.COLOR_NIGHTMARE_GLOW, 1.5)
                        if nightmare in self.nightmares:
                            self.nightmares.remove(nightmare)
                        reward += 1.0 # Reward for destroying nightmare
                        # sfx: nightmare_death.wav
                        break # Nightmare is gone, stop checking blocks
            
            if collided_with_block:
                nightmare["pos"] -= direction * self.NIGHTMARE_SPEED # Stop movement if blocked

        return reward
    
    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.05 # Gravity
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        # This method is for internal logic, the step method will return the correct flags
        return self.city_health <= 0 or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "city_health": self.city_health,
            "blocks_placed": len(self.blocks),
            "nightmares_active": len(self.nightmares),
        }

    def _get_observation(self):
        self.screen.blit(self._background_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _create_gradient_background(self):
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_START[0] * (1 - ratio) + self.COLOR_BG_END[0] * ratio),
                int(self.COLOR_BG_START[1] * (1 - ratio) + self.COLOR_BG_END[1] * ratio),
                int(self.COLOR_BG_START[2] * (1 - ratio) + self.COLOR_BG_END[2] * ratio)
            )
            pygame.draw.line(bg, color, (0, y), (self.SCREEN_WIDTH, y))
        return bg

    def _render_game(self):
        # Foundation
        glow_rect = self.foundation_rect.inflate(10, 10)
        pygame.draw.rect(self.screen, self.COLOR_FOUNDATION_GLOW, glow_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.COLOR_FOUNDATION, self.foundation_rect, border_radius=8)

        # Blocks
        for block in self.blocks:
            info = self.BLOCK_TYPES[block["type"]]
            glow_rect = block["rect"].inflate(8, 8)
            pygame.draw.rect(self.screen, info["glow"], glow_rect, border_radius=6)
            pygame.draw.rect(self.screen, info["color"], block["rect"], border_radius=4)
            # Health bar
            health_ratio = block["health"] / block["max_health"]
            health_bar_rect = pygame.Rect(block["rect"].left, block["rect"].top - 7, block["rect"].width * health_ratio, 4)
            pygame.draw.rect(self.screen, (0, 255, 0), health_bar_rect)
        
        # Nightmares
        for n in self.nightmares:
            pos = (int(n["pos"][0]), int(n["pos"][1]))
            radius = int(n["radius"] * (1 + 0.1 * math.sin(self.steps * 0.2)))
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(radius * 1.5), self.COLOR_NIGHTMARE_GLOW)
            # Core
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_NIGHTMARE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_NIGHTMARE)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / p["max_life"]))))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["size"]), int(p["pos"][1] - p["size"])))

        # Cursor
        block_name = self.unlocked_block_types[self.current_block_type_idx]
        block_info = self.BLOCK_TYPES[block_name]
        block_size = block_info["size"]
        grid_x = round((self.cursor_pos[0] - block_size[0] / 2) / self.GRID_SIZE) * self.GRID_SIZE
        grid_y = round((self.cursor_pos[1] - block_size[1] / 2) / self.GRID_SIZE) * self.GRID_SIZE
        cursor_rect = pygame.Rect(grid_x, grid_y, block_size[0], block_size[1])
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2, border_radius=4)

    def _render_ui(self):
        # --- Top Left: Score and Time ---
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        time_text = self.font_small.render(f"DAWN IN: {self.MAX_STEPS - self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (10, 30))

        # --- Top Right: Block Info ---
        block_name = self.unlocked_block_types[self.current_block_type_idx].upper()
        block_text = self.font_small.render(f"BLOCK: {block_name}", True, self.COLOR_TEXT)
        self.screen.blit(block_text, (self.SCREEN_WIDTH - block_text.get_width() - 10, 10))

        # --- Bottom Center: City Health Bar ---
        health_percent = max(0, self.city_health / self.MAX_CITY_HEALTH)
        bar_width = 300
        bar_height = 20
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = self.SCREEN_HEIGHT - bar_height - 10

        # Background bar
        pygame.draw.rect(self.screen, self.COLOR_FOUNDATION_GLOW, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        # Health fill
        fill_width = bar_width * health_percent
        fill_color = (0, 255, 0) if health_percent > 0.5 else ((255, 255, 0) if health_percent > 0.2 else (255, 0, 0))
        pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, fill_width, bar_height), border_radius=5)
        # Text on bar
        health_label = self.font_small.render("CITY HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_label, (bar_x + bar_width/2 - health_label.get_width()/2, bar_y + bar_height/2 - health_label.get_height()/2))

        # --- Game Over/Win Text ---
        if self._check_termination():
            if self.steps >= self.MAX_STEPS:
                end_text_str = "DAWN ARRIVES. YOU SURVIVED."
                color = self.COLOR_UI_ACCENT
            else:
                end_text_str = "THE CITY HAS FALLEN."
                color = self.COLOR_NIGHTMARE
            end_text = self.font_large.render(end_text_str, True, color)
            pos = (self.SCREEN_WIDTH/2 - end_text.get_width()/2, self.SCREEN_HEIGHT/2 - end_text.get_height()/2)
            self.screen.blit(end_text, pos)

    def _create_particles(self, pos, count, color, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "max_life": 30,
                "color": color,
                "size": self.np_random.integers(2, 5)
            })

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Example Usage & Manual Play ---
    # This block needs a display, so we unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Setup for manual play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Dream City Defense")
    clock = pygame.time.Clock()
    
    running = True
    done = False
    
    while running:
        if done:
            # Wait for a key press to reset after game over
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    obs, info = env.reset()
                    done = False
        else:
            # --- Action mapping for human player ---
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # --- Rendering ---
            # The observation is the rendered frame, so we just need to display it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(GameEnv.FPS)

    env.close()