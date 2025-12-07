
# Generated: 2025-08-28T06:19:26.547073
# Source Brief: brief_02899.md
# Brief Index: 2899

        
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
        "Controls: Use arrow keys (↑↓←→) to navigate the cavern."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated isometric cavern, collecting all 15 crystals while avoiding deadly traps to achieve a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 20, 20
        self.NUM_CRYSTALS = 15
        self.NUM_TRAPS = 30
        self.MAX_STEPS = 1000
        
        # Isometric projection constants
        self.TILE_W = 32
        self.TILE_H = 16
        self.TILE_W_HALF = self.TILE_W // 2
        self.TILE_H_HALF = self.TILE_H // 2
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_TILE = (40, 45, 65)
        self.COLOR_TILE_EDGE = (55, 60, 85)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 255, 255, 50)
        self.COLOR_CRYSTAL_OUTER = (255, 0, 255)
        self.COLOR_CRYSTAL_INNER = (255, 255, 255)
        self.COLOR_TRAP_OUTER = (80, 20, 20)
        self.COLOR_TRAP_INNER_BASE = (200, 30, 30)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_BG = (30, 35, 50, 200)

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = (0, 0)
        self.crystal_locs = []
        self.trap_locs = []
        self.crystals_collected = 0
        self.particles = []
        self.termination_reason = ""
        self.np_random = None

        # This call is for dev; remove if it causes issues in production environments.
        # self.validate_implementation()
    
    def _iso_to_cart(self, iso_x, iso_y):
        """Converts isometric grid coordinates to cartesian pixel coordinates."""
        cart_x = self.ORIGIN_X + (iso_x - iso_y) * self.TILE_W_HALF
        cart_y = self.ORIGIN_Y + (iso_x + iso_y) * self.TILE_H_HALF
        return int(cart_x), int(cart_y)

    def _generate_level(self):
        """Generates a new level with player, crystals, and traps."""
        all_coords = set((x, y) for x in range(self.GRID_W) for y in range(self.GRID_H))

        # Place player at the bottom-center
        self.player_pos = (self.GRID_W // 2, self.GRID_H - 1)
        occupied_coords = {self.player_pos}

        # Create a safe zone around the player's start
        safe_zone = set()
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                px, py = self.player_pos
                if 0 <= px + dx < self.GRID_W and 0 <= py + dy < self.GRID_H:
                    safe_zone.add((px + dx, py + dy))

        # Generate a path to ensure crystals are reachable
        path = set()
        start_node = (self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H))
        path.add(start_node)
        
        # Use a random walk to create a connected path
        for _ in range(150): # Longer walk for better coverage
            current_pos = random.choice(list(path))
            moves = [(-1,0), (1,0), (0,-1), (0,1)]
            self.np_random.shuffle(moves)
            for dx, dy in moves:
                nx, ny = current_pos[0] + dx, current_pos[1] + dy
                if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H:
                    path.add((nx,ny))
                    break
        
        # Place crystals on the path, avoiding the player's start
        available_for_crystals = list(path - occupied_coords)
        if len(available_for_crystals) < self.NUM_CRYSTALS: # Fallback if path is too small
            available_for_crystals.extend(list(all_coords - path - occupied_coords))
        
        crystal_indices = self.np_random.choice(len(available_for_crystals), self.NUM_CRYSTALS, replace=False)
        self.crystal_locs = [available_for_crystals[i] for i in crystal_indices]
        occupied_coords.update(self.crystal_locs)
        
        # Place traps in remaining empty spaces, avoiding the safe zone
        available_for_traps = list(all_coords - occupied_coords - safe_zone)
        trap_indices = self.np_random.choice(len(available_for_traps), min(len(available_for_traps), self.NUM_TRAPS), replace=False)
        self.trap_locs = [available_for_traps[i] for i in trap_indices]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.crystals_collected = 0
        self.particles = []
        self.termination_reason = ""
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()
    
    def _get_dist_to_nearest_crystal(self, pos):
        if not self.crystal_locs:
            return None
        px, py = pos
        return min(math.hypot(px - cx, py - cy) for cx, cy in self.crystal_locs)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        terminated = False
        
        # Unpack factorized action
        movement = action[0]
        
        old_pos = self.player_pos
        old_dist = self._get_dist_to_nearest_crystal(old_pos)
        
        # Update player position based on movement action
        px, py = self.player_pos
        if movement == 1 and py > 0: py -= 1  # Up
        elif movement == 2 and py < self.GRID_H - 1: py += 1  # Down
        elif movement == 3 and px > 0: px -= 1  # Left
        elif movement == 4 and px < self.GRID_W - 1: px += 1  # Right
        self.player_pos = (px, py)
        
        # Calculate movement reward
        new_dist = self._get_dist_to_nearest_crystal(self.player_pos)
        if old_dist is not None and new_dist is not None:
            if new_dist < old_dist:
                reward += 1.0  # Closer to a crystal
            else:
                reward -= 0.1 # Moved away or stayed same distance
        
        # Check for events
        if self.player_pos in self.crystal_locs:
            # Sound: crystal_collect.wav
            reward += 10.0
            self.crystal_locs.remove(self.player_pos)
            self.crystals_collected += 1
            self._spawn_particles(self.player_pos, self.COLOR_CRYSTAL_OUTER)
        
        if self.player_pos in self.trap_locs:
            # Sound: trap_spring.wav
            reward -= 100.0
            terminated = True
            self.game_over = True
            self.termination_reason = "You fell into a trap!"
            self._spawn_particles(self.player_pos, self.COLOR_TRAP_INNER_BASE, 30)

        # Check for win/termination conditions
        if not self.crystal_locs:
            # Sound: level_complete.wav
            reward += 100.0
            terminated = True
            self.game_over = True
            self.termination_reason = "All crystals collected!"
            
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.termination_reason = "Maximum steps reached."

        self.score += reward
        self._update_particles()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _spawn_particles(self, pos, color, count=20):
        cx, cy = self._iso_to_cart(*pos)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(20, 40)
            size = self.np_random.uniform(3, 6)
            self.particles.append([ [cx, cy], vel, life, size, color ])

    def _update_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0] # pos.x += vel.x
            p[0][1] += p[1][1] # pos.y += vel.y
            p[2] -= 1          # life -= 1
            p[3] *= 0.95       # size shrinks
        self.particles = [p for p in self.particles if p[2] > 0 and p[3] > 0.5]
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render floor tiles
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                cx, cy = self._iso_to_cart(x, y)
                points = [
                    (cx, cy - self.TILE_H_HALF),
                    (cx + self.TILE_W_HALF, cy),
                    (cx, cy + self.TILE_H_HALF),
                    (cx - self.TILE_W_HALF, cy)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_TILE)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_TILE_EDGE)

        # Render traps
        pulse = (math.sin(self.steps * 0.15) + 1) / 2 # Smooth 0-1 pulse
        pulse_color = tuple(int(c1 + (c2 - c1) * pulse) for c1, c2 in zip(self.COLOR_TRAP_OUTER, self.COLOR_TRAP_INNER_BASE))
        for tx, ty in self.trap_locs:
            cx, cy = self._iso_to_cart(tx, ty)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, self.TILE_W_HALF // 2, self.COLOR_TRAP_OUTER)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, int(self.TILE_W_HALF // 2.5), pulse_color)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, self.TILE_W_HALF // 2, self.COLOR_TRAP_OUTER)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, int(self.TILE_W_HALF // 2.5), pulse_color)

        # Render crystals
        for crx, cry in self.crystal_locs:
            cx, cy = self._iso_to_cart(crx, cry)
            size = self.TILE_H_HALF * 0.8
            angle = (self.steps * 0.02) % (2 * math.pi)
            points = []
            for i in range(6):
                a = angle + i * (2 * math.pi / 6)
                points.append((cx + math.cos(a) * size, cy - self.TILE_H_HALF * 0.5 + math.sin(a) * size * 0.5))
            
            glow_radius = int(size * (1.5 + 0.2 * math.sin(self.steps * 0.1 + crx)))
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_CRYSTAL_OUTER, 30), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (cx - glow_radius, cy - self.TILE_H_HALF * 0.5 - glow_radius))

            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CRYSTAL_OUTER)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CRYSTAL_INNER)

        # Render player
        px, py = self._iso_to_cart(*self.player_pos)
        player_size = self.TILE_H_HALF
        
        glow_surf = pygame.Surface((player_size * 4, player_size * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (player_size*2, player_size*2), player_size*2)
        self.screen.blit(glow_surf, (px - player_size*2, py - player_size*2 - player_size//2))
        
        player_points = [
            (px, py - player_size),
            (px + player_size * 0.7, py),
            (px, py),
            (px - player_size * 0.7, py),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, player_points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, player_points, (255,255,255))
        
        # Render particles
        for p in self.particles:
            pos, _, life, size, color = p
            alpha = int(255 * (life / 40))
            if size > 1:
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(size), (*color, alpha))
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), int(size), (*color, alpha))

    def _render_ui(self):
        # UI Panel
        ui_rect = pygame.Rect(10, 10, 220, 40)
        pygame.gfxdraw.box(self.screen, ui_rect, self.COLOR_UI_BG)
        
        # Crystal count
        crystal_text = f"Crystals: {self.crystals_collected} / {self.NUM_CRYSTALS}"
        text_surf = self.font_ui.render(crystal_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (20, 18))
        
        # Score
        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        score_rect = score_surf.get_rect(topright=(self.WIDTH - 20, 18))
        pygame.gfxdraw.box(self.screen, score_rect.inflate(20, 12), self.COLOR_UI_BG)
        self.screen.blit(score_surf, score_rect)
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            reason_surf = self.font_game_over.render(self.termination_reason, True, self.COLOR_UI_TEXT)
            reason_rect = reason_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(reason_surf, reason_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "crystals_collected": self.crystals_collected,
            "player_pos": self.player_pos,
        }
    
    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
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


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Crystal Caverns")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        # Since auto_advance is False, we only step on a key press
        # or can add a delay to step automatically
        action = [movement, 0, 0] # Space and Shift are not used
        
        # For human play, we want to step every frame a key is held
        # This deviates from the pure turn-based RL model but is better for playability
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # Wait for a moment then reset
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0
            print("--- Game Reset ---")

        clock.tick(10) # Control human play speed

    env.close()