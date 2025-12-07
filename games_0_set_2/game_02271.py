
# Generated: 2025-08-28T04:16:58.725604
# Source Brief: brief_02271.md
# Brief Index: 2271

        
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
    """
    A top-down survival game where the player must escape hordes of zombies to reach an exit.
    The game is turn-based, with difficulty increasing across three stages within a single episode.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your character. Avoid the blue zombies and reach the green exit."
    )

    game_description = (
        "Survive the zombie horde! Navigate through increasingly difficult stages by evading zombies and reaching the exit before the timer runs out."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 32, 20
        self.TILE_SIZE = self.WIDTH // self.GRID_W
        self.TOTAL_STAGES = 3
        self.MAX_TIME_PER_STAGE = 60

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_PLAYER = (255, 0, 80)
        self.COLOR_ZOMBIE = (0, 150, 255)
        self.COLOR_EXIT = (0, 255, 120)
        self.COLOR_BOUNDARY = (0, 0, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PARTICLE = (255, 255, 255)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        try:
            self.font_large = pygame.font.SysFont("Consolas", 24)
            self.font_small = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 28)
            self.font_small = pygame.font.Font(None, 22)

        # --- Persistent State ---
        # This state persists across resets until the full game is won.
        self.max_stages_completed = 0

        # --- Game State (initialized in reset) ---
        self.player_pos = None
        self.exit_pos = None
        self.zombies = None
        self.particles = None
        self.current_stage = None
        self.timer = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.stage_configs = None

        # Initialize state variables and validate implementation
        self.reset()
        self.validate_implementation()

    def _setup_stage(self):
        """Initializes the game state for the current stage."""
        self.timer = self.MAX_TIME_PER_STAGE
        self.player_pos = pygame.Vector2(self.GRID_W // 2, self.GRID_H - 2)
        self.exit_pos = pygame.Vector2(self.GRID_W // 2, 1)
        self.zombies = []
        self.particles = []

        # Define stage configurations
        self.stage_configs = {
            1: {"zombie_count": 10, "zombie_speed": 0.5},
            2: {"zombie_count": 12, "zombie_speed": 0.55},
            3: {"zombie_count": 14, "zombie_speed": 0.6},
        }
        config = self.stage_configs[self.current_stage]
        
        # Generate valid spawn points
        possible_spawns = [pygame.Vector2(x, y) for x in range(self.GRID_W) for y in range(self.GRID_H)]
        if self.player_pos in possible_spawns:
            possible_spawns.remove(self.player_pos)
        if self.exit_pos in possible_spawns:
            possible_spawns.remove(self.exit_pos)
        
        # Spawn zombies
        spawn_indices = self.np_random.choice(len(possible_spawns), size=config["zombie_count"], replace=False)
        for i in spawn_indices:
            self.zombies.append({"pos": pygame.Vector2(possible_spawns[i]), "speed": config["zombie_speed"]})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.max_stages_completed >= self.TOTAL_STAGES:
            self.max_stages_completed = 0  # Reset progress after a full win

        self.current_stage = self.max_stages_completed + 1
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        self._setup_stage()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0
        terminated = False

        self.steps += 1
        self.timer -= 1

        # --- 1. Player Movement & Reward ---
        prev_player_pos = self.player_pos.copy()
        if movement == 1:  # Up
            self.player_pos.y -= 1
        elif movement == 2:  # Down
            self.player_pos.y += 1
        elif movement == 3:  # Left
            self.player_pos.x -= 1
        elif movement == 4:  # Right
            self.player_pos.x += 1
        
        # Clamp player position to be within bounds
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.GRID_W - 1)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.GRID_H - 1)

        # Reward for surviving and moving closer to exit
        reward += 0.1  # Survival reward
        prev_dist = prev_player_pos.distance_to(self.exit_pos)
        new_dist = self.player_pos.distance_to(self.exit_pos)
        if new_dist > prev_dist and movement != 0:
            reward -= 0.2 # Penalty for moving away

        # --- 2. Zombie Movement ---
        for zombie in self.zombies:
            z_pos = zombie["pos"]
            p_pos_int = pygame.Vector2(int(self.player_pos.x), int(self.player_pos.y))
            
            direction = p_pos_int - z_pos
            if direction.length() > 0:
                # Move horizontally or vertically, whichever is greater distance
                if abs(direction.x) > abs(direction.y):
                    z_pos.x += np.sign(direction.x) * zombie["speed"]
                else:
                    z_pos.y += np.sign(direction.y) * zombie["speed"]

        # --- 3. Particle Update ---
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

        # --- 4. Check Win/Loss Conditions ---
        # Reached Exit
        if self.player_pos == self.exit_pos:
            self.max_stages_completed = max(self.max_stages_completed, self.current_stage)
            if self.current_stage >= self.TOTAL_STAGES:
                reward += 100  # Final win
                terminated = True
                self.game_over = True
            else:
                reward += 5  # Stage complete
                self.current_stage += 1
                self._setup_stage() # Load next stage, continue episode
        
        # Zombie Collision
        for zombie in self.zombies:
            if pygame.Vector2(int(zombie["pos"].x), int(zombie["pos"].y)) == self.player_pos:
                reward -= 100
                terminated = True
                self.game_over = True
                self._create_death_particles()
                # sfx: player_death_explosion
                break
        
        # Timeout
        if self.timer <= 0 and not terminated:
            reward -= 50
            terminated = True
            self.game_over = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )
    
    def _create_death_particles(self):
        """Generates particles for the player's death animation."""
        px, py = self._grid_to_pixel(self.player_pos)
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                "pos": pygame.Vector2(px, py),
                "vel": vel,
                "lifespan": self.np_random.integers(10, 20),
                "radius": self.np_random.uniform(1, 3)
            })

    def _grid_to_pixel(self, grid_pos):
        """Converts grid coordinates to pixel coordinates for rendering."""
        px = grid_pos.x * self.TILE_SIZE + self.TILE_SIZE / 2
        py = grid_pos.y * self.TILE_SIZE + self.TILE_SIZE / 2
        return int(px), int(py)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw boundaries (a subtle grid)
        for x in range(0, self.WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.screen, (30, 30, 40), (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.screen, (30, 30, 40), (0, y), (self.WIDTH, y), 1)

        # Draw Exit
        exit_px, exit_py = self._grid_to_pixel(self.exit_pos)
        exit_rect = pygame.Rect(exit_px - self.TILE_SIZE // 2, exit_py - self.TILE_SIZE // 2, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)
        pygame.draw.rect(self.screen, tuple(c*0.8 for c in self.COLOR_EXIT), exit_rect, 3)

        # Draw Zombies
        for zombie in self.zombies:
            z_px, z_py = self._grid_to_pixel(zombie["pos"])
            size = int(self.TILE_SIZE * 0.8)
            zombie_rect = pygame.Rect(z_px - size // 2, z_py - size // 2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, zombie_rect, border_radius=3)

        # Draw Player
        if not self.game_over or self.timer <= 0:
            p_px, p_py = self._grid_to_pixel(self.player_pos)
            radius = int(self.TILE_SIZE * 0.4)
            # Glow effect
            glow_radius = int(radius * 2.5)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, self.COLOR_PLAYER + (30,))
            self.screen.blit(glow_surf, (p_px - glow_radius, p_py - glow_radius))
            # Player circle
            pygame.gfxdraw.filled_circle(self.screen, p_px, p_py, radius, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, p_px, p_py, radius, self.COLOR_PLAYER)

        # Draw Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, self.COLOR_PARTICLE, p["pos"], p["radius"])
            
    def _render_ui(self):
        # Render Stage
        stage_text = self.font_large.render(f"STAGE {self.current_stage}/{self.TOTAL_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH - stage_text.get_width() - 10, 10))

        # Render Timer
        timer_color = self.COLOR_TEXT if self.timer > 10 else (255, 100, 100)
        timer_text = self.font_large.render(f"TIME: {self.timer}", True, timer_color)
        self.screen.blit(timer_text, (10, 10))
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "timer": self.timer
        }

    def close(self):
        pygame.font.quit()
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Zombie Survival")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    print(env.user_guide)
    
    while not terminated:
        # --- Human Input ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        action = [movement, 0, 0] # Space and Shift are not used

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # Convert the observation (H, W, C) back to a surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # Since auto_advance is False, we control the step rate
        clock.tick(10) # Run at 10 steps per second for human playability
        
    print(f"Game Over! Final Score: {total_reward:.2f}, Info: {info}")
    env.close()