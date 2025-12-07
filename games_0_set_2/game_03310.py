
# Generated: 2025-08-27T22:58:39.746369
# Source Brief: brief_03310.md
# Brief Index: 3310

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move in isometric directions. Goal: Collect 100 crystals."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated isometric cavern, collecting crystals while avoiding perilous pitfalls to amass a shimmering fortune."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.WORLD_SIZE = 60
        self.NUM_CRYSTALS = 150
        self.NUM_PITS = 250
        self.WIN_SCORE = 100
        self.MAX_STEPS = 1000
        self.TILE_WIDTH_H, self.TILE_HEIGHT_H = 28, 14 # Half-sizes

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Visuals
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 60, 80)
        self.COLOR_PIT = (0, 0, 0)
        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_PLAYER_GLOW = (255, 80, 80, 50)
        self.COLOR_CRYSTAL = (255, 255, 0)
        self.COLOR_CRYSTAL_GLOW = (255, 255, 100, 50)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)

        # State variables (initialized in reset)
        self.player_pos = None
        self.world_pits = None
        self.world_crystals = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.particles = []
        
        # Initialize state variables
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Loop until a valid, winnable world is generated
        while True:
            self._generate_world()
            if self._is_world_valid():
                break

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _generate_world(self):
        self.player_pos = (self.WORLD_SIZE // 2, self.WORLD_SIZE // 2)
        
        self.world_pits = set()
        self.world_crystals = set()

        start_area_radius = 4
        
        # Generate pits
        for _ in range(self.NUM_PITS):
            while True:
                pos = (self.np_random.integers(0, self.WORLD_SIZE), self.np_random.integers(0, self.WORLD_SIZE))
                dist_to_start = abs(pos[0] - self.player_pos[0]) + abs(pos[1] - self.player_pos[1])
                if pos != self.player_pos and dist_to_start > start_area_radius:
                    self.world_pits.add(pos)
                    break
        
        # Generate crystals
        for _ in range(self.NUM_CRYSTALS):
            while True:
                pos = (self.np_random.integers(0, self.WORLD_SIZE), self.np_random.integers(0, self.WORLD_SIZE))
                if pos != self.player_pos and pos not in self.world_pits:
                    self.world_crystals.add(pos)
                    break
    
    def _is_world_valid(self):
        q = deque([self.player_pos])
        visited = {self.player_pos}
        reachable_crystals = 0
        
        while q:
            x, y = q.popleft()
            
            if (x, y) in self.world_crystals:
                reachable_crystals += 1
            
            if reachable_crystals >= self.WIN_SCORE:
                return True

            # Standard cartesian neighbors for grid traversal
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                if 0 <= nx < self.WORLD_SIZE and 0 <= ny < self.WORLD_SIZE and neighbor not in visited and neighbor not in self.world_pits:
                    visited.add(neighbor)
                    q.append(neighbor)
        return False

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        terminated = False

        # --- Update Game Logic ---
        self.steps += 1
        
        # Isometric movement mapping
        dx, dy = 0, 0
        if movement == 1: # Up -> Up-Left
            dx, dy = -1, 0
        elif movement == 2: # Down -> Down-Right
            dx, dy = 1, 0
        elif movement == 3: # Left -> Down-Left
            dx, dy = 0, 1
        elif movement == 4: # Right -> Up-Right
            dx, dy = 0, -1
        
        if dx != 0 or dy != 0:
            next_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            
            # World boundary check
            if not (0 <= next_pos[0] < self.WORLD_SIZE and 0 <= next_pos[1] < self.WORLD_SIZE):
                next_pos = self.player_pos # Stay put if hitting boundary

            # Collision detection
            if next_pos in self.world_pits:
                reward = -100
                terminated = True
                self.game_over = True
                # Player position does not update, stays on edge of pit
            else:
                self.player_pos = next_pos
                if self.player_pos in self.world_crystals:
                    # Sound: Crystal collect
                    self.world_crystals.remove(self.player_pos)
                    self.score += 1
                    reward = 1
                    self._spawn_particles(self.player_pos, self.COLOR_CRYSTAL, 20)
                    if self.score >= self.WIN_SCORE:
                        # Sound: Win fanfare
                        reward += 100
                        terminated = True
                        self.game_over = True
                        self.win_state = True

        # Check for max steps termination
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over = True

        self._update_particles()
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _grid_to_screen(self, grid_x, grid_y):
        # Center view on player
        rel_x = grid_x - self.player_pos[0]
        rel_y = grid_y - self.player_pos[1]
        
        screen_x = self.SCREEN_WIDTH // 2 + (rel_x - rel_y) * self.TILE_WIDTH_H
        screen_y = self.SCREEN_HEIGHT // 2 + (rel_x + rel_y) * self.TILE_HEIGHT_H
        return int(screen_x), int(screen_y)

    def _render_tile(self, screen_pos, color, border_color):
        points = [
            (screen_pos[0], screen_pos[1] - self.TILE_HEIGHT_H),
            (screen_pos[0] + self.TILE_WIDTH_H, screen_pos[1]),
            (screen_pos[0], screen_pos[1] + self.TILE_HEIGHT_H),
            (screen_pos[0] - self.TILE_WIDTH_H, screen_pos[1]),
        ]
        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.aalines(self.screen, border_color, True, points, 1)

    def _render_crystal(self, screen_pos):
        pulse = math.sin(pygame.time.get_ticks() * 0.005) * 2
        size_h = self.TILE_HEIGHT_H * 0.6 + pulse
        size_w = self.TILE_WIDTH_H * 0.6 + pulse
        
        points = [
            (screen_pos[0], screen_pos[1] - size_h),
            (screen_pos[0] + size_w, screen_pos[1]),
            (screen_pos[0], screen_pos[1] + size_h),
            (screen_pos[0] - size_w, screen_pos[1]),
        ]
        # Glow
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CRYSTAL_GLOW)
        # Crystal
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CRYSTAL)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CRYSTAL)

    def _spawn_particles(self, grid_pos, color, count):
        screen_pos = self._grid_to_screen(grid_pos[0], grid_pos[1])
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(15, 30)
            self.particles.append({'pos': list(screen_pos), 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _render_game(self):
        # Determine visible grid range
        view_w = self.SCREEN_WIDTH // (self.TILE_WIDTH_H * 2) + 4
        view_h = self.SCREEN_HEIGHT // (self.TILE_HEIGHT_H * 2) + 4
        
        min_gx = int(self.player_pos[0] - view_w)
        max_gx = int(self.player_pos[0] + view_w)
        min_gy = int(self.player_pos[1] - view_h)
        max_gy = int(self.player_pos[1] + view_h)
        
        # Painter's algorithm: draw from far to near
        render_list = []
        for gy in range(min_gy, max_gy):
            for gx in range(min_gx, max_gx):
                if not (0 <= gx < self.WORLD_SIZE and 0 <= gy < self.WORLD_SIZE):
                    continue
                
                screen_pos = self._grid_to_screen(gx, gy)
                
                # Cull off-screen tiles
                if not (-self.TILE_WIDTH_H < screen_pos[0] < self.SCREEN_WIDTH + self.TILE_WIDTH_H and \
                        -self.TILE_HEIGHT_H < screen_pos[1] < self.SCREEN_HEIGHT + self.TILE_HEIGHT_H):
                    continue
                
                # Add to render list with depth
                depth = gx + gy
                render_list.append((depth, gx, gy, screen_pos))
        
        # Sort by depth and render
        render_list.sort(key=lambda item: item[0])

        for _, gx, gy, screen_pos in render_list:
            pos = (gx, gy)
            if pos in self.world_pits:
                self._render_tile(screen_pos, self.COLOR_PIT, self.COLOR_PIT)
            else:
                self._render_tile(screen_pos, self.COLOR_BG, self.COLOR_GRID)
                if pos in self.world_crystals:
                    self._render_crystal(screen_pos)

        # Render particles
        for p in self.particles:
            size = max(1, p['life'] / 6)
            pygame.draw.circle(self.screen, p['color'], p['pos'], size)
        
        # Render player
        player_screen_pos = self._grid_to_screen(self.player_pos[0], self.player_pos[1])
        size_h, size_w = self.TILE_HEIGHT_H * 0.8, self.TILE_WIDTH_H * 0.8
        points = [
            (player_screen_pos[0], player_screen_pos[1] - size_h),
            (player_screen_pos[0] + size_w, player_screen_pos[1]),
            (player_screen_pos[0], player_screen_pos[1] + size_h),
            (player_screen_pos[0] - size_w, player_screen_pos[1]),
        ]
        # Glow
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER_GLOW)
        # Player
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score display
        score_text = self.font_ui.render(f"CRYSTALS: {self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Steps display
        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            if self.win_state:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_LOSE
            
            text = self.font_game_over.render(msg, True, color)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(text, text_rect)

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
            "player_pos": self.player_pos,
            "win": self.win_state,
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Override metadata for human rendering
    env.metadata["render_modes"] = ["human"]
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Crystal Caverns")
    clock = pygame.time.Clock()

    print(env.user_guide)
    print(env.game_description)

    action = env.action_space.sample()
    action[0] = 0 # Start with no-op

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        mov = 0 # No-op
        if keys[pygame.K_UP]:
            mov = 1
        elif keys[pygame.K_DOWN]:
            mov = 2
        elif keys[pygame.K_LEFT]:
            mov = 3
        elif keys[pygame.K_RIGHT]:
            mov = 4
        
        action = [mov, 0, 0] # Space and shift not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the display window
        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(draw_surface, (0, 0))
        pygame.display.flip()
        
        if done:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(3000) # Wait 3 seconds before closing
        
        # Since auto_advance is False, we control the step rate
        clock.tick(10) # Limit to 10 actions per second for playability

    env.close()