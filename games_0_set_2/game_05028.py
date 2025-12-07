
# Generated: 2025-08-28T03:45:21.135342
# Source Brief: brief_05028.md
# Brief Index: 5028

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your character on the grid. Collect all donuts!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect all 20 donuts before the time runs out, but watch out for the red obstacles!"
    )

    # Should frames auto-advance or wait for user input?
    # The brief specifies "Each action represents one turn", so auto_advance is False.
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 32, 20
        self.CELL_SIZE = 20
        
        self.NUM_DONUTS = 20
        self.NUM_OBSTACLES = 3
        self.TIME_LIMIT_STEPS = 1000
        
        # Colors (Retro Pixel Art)
        self.COLOR_BG = (20, 20, 40)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_PLAYER = (255, 255, 0) # Yellow
        self.COLOR_DONUT = (255, 105, 180) # Pink
        self.COLOR_OBSTACLE = (255, 0, 0) # Red
        self.COLOR_TEXT = (255, 255, 255) # White
        self.COLOR_FLASH = (255, 0, 0, 100) # Red flash for collision
        
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
        try:
            self.font_large = pygame.font.SysFont("Consolas", 24)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 30)

        # --- Game State Initialization ---
        self.player_pos = (0, 0)
        self.donuts = []
        self.obstacles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.effects = []
        
        self.reset()
        
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.effects = []
        
        self._spawn_entities()

        return self._get_observation(), self._get_info()

    def _spawn_entities(self):
        """Places the player, donuts, and obstacles on the grid without overlap."""
        all_coords = set((x, y) for x in range(self.GRID_W) for y in range(self.GRID_H))

        # Player in center
        self.player_pos = (self.GRID_W // 2, self.GRID_H // 2)
        all_coords.remove(self.player_pos)
        
        # Don't spawn obstacles right next to player
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                neighbor = (self.player_pos[0] + dx, self.player_pos[1] + dy)
                if neighbor in all_coords:
                    all_coords.remove(neighbor)

        # Obstacles
        obstacle_coords = self.np_random.choice(list(all_coords), size=self.NUM_OBSTACLES, replace=False)
        self.obstacles = [tuple(pos) for pos in obstacle_coords]
        for pos in self.obstacles:
            all_coords.add(tuple(pos)) # Re-add non-neighbor cells
        for pos in self.obstacles:
            all_coords.remove(tuple(pos))

        # Donuts
        donut_coords = self.np_random.choice(list(all_coords), size=self.NUM_DONUTS, replace=False)
        self.donuts = [tuple(pos) for pos in donut_coords]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        self.steps += 1
        
        old_player_pos = self.player_pos
        dist_before = self._get_dist_to_nearest_donut(old_player_pos)
        
        # --- Player Movement ---
        px, py = self.player_pos
        if movement == 1: py -= 1 # Up
        elif movement == 2: py += 1 # Down
        elif movement == 3: px -= 1 # Left
        elif movement == 4: px += 1 # Right
        
        # Boundary checks
        px = max(0, min(self.GRID_W - 1, px))
        py = max(0, min(self.GRID_H - 1, py))
        self.player_pos = (px, py)
        
        # --- Reward for Movement ---
        dist_after = self._get_dist_to_nearest_donut(self.player_pos)
        if dist_after < dist_before:
            reward += 1.0
        elif dist_after > dist_before:
            reward -= 0.1
            
        # --- Collision & Event Handling ---
        if self.player_pos in self.donuts:
            self.donuts.remove(self.player_pos)
            self.score += 1
            reward += 10.0
            # SFX: Donut collect sound
            self._add_effect('sparkle', self.player_pos)

        if self.player_pos in self.obstacles:
            self.game_over = True
            reward = -50.0 # Override other rewards
            # SFX: Crash sound
            self._add_effect('flash', (0,0))

        # --- Termination Conditions ---
        terminated = False
        if self.game_over:
            terminated = True
            
        if not self.donuts: # Win
            terminated = True
            self.game_over = True
            reward += 50.0
            # SFX: Win fanfare
            
        if self.steps >= self.TIME_LIMIT_STEPS: # Time out
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_dist_to_nearest_donut(self, pos):
        """Calculates Manhattan distance to the nearest donut."""
        if not self.donuts:
            return 0
        px, py = pos
        return min(abs(px - dx) + abs(py - dy) for dx, dy in self.donuts)

    def _add_effect(self, effect_type, pos):
        self.effects.append({'type': effect_type, 'pos': pos})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        grid_surface = pygame.Surface((self.GRID_W * self.CELL_SIZE, self.GRID_H * self.CELL_SIZE))
        grid_surface.fill(self.COLOR_BG)

        # --- Draw Grid ---
        for x in range(self.GRID_W + 1):
            start_pos = (x * self.CELL_SIZE, 0)
            end_pos = (x * self.CELL_SIZE, self.GRID_H * self.CELL_SIZE)
            pygame.draw.line(grid_surface, self.COLOR_GRID, start_pos, end_pos)
        for y in range(self.GRID_H + 1):
            start_pos = (0, y * self.CELL_SIZE)
            end_pos = (self.GRID_W * self.CELL_SIZE, y * self.CELL_SIZE)
            pygame.draw.line(grid_surface, self.COLOR_GRID, start_pos, end_pos)

        # --- Draw Obstacles ---
        for ox, oy in self.obstacles:
            rect = pygame.Rect(ox * self.CELL_SIZE, oy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(grid_surface, self.COLOR_OBSTACLE, rect)

        # --- Draw Donuts ---
        for dx, dy in self.donuts:
            center = (int((dx + 0.5) * self.CELL_SIZE), int((dy + 0.5) * self.CELL_SIZE))
            radius = int(self.CELL_SIZE * 0.4)
            pygame.draw.circle(grid_surface, self.COLOR_DONUT, center, radius)
            pygame.draw.circle(grid_surface, self.COLOR_BG, center, int(radius * 0.4))

        # --- Draw Player ---
        px, py = self.player_pos
        center = (int((px + 0.5) * self.CELL_SIZE), int((py + 0.5) * self.CELL_SIZE))
        pygame.draw.circle(grid_surface, self.COLOR_PLAYER, center, int(self.CELL_SIZE * 0.45))
        
        self.screen.blit(grid_surface, (0,0))
        
        # --- Draw Effects ---
        for effect in self.effects:
            if effect['type'] == 'sparkle':
                ex, ey = effect['pos']
                center = (int((ex + 0.5) * self.CELL_SIZE), int((ey + 0.5) * self.CELL_SIZE))
                for i in range(8):
                    angle = i * (math.pi / 4)
                    start = (center[0] + math.cos(angle) * (self.CELL_SIZE * 0.2), center[1] + math.sin(angle) * (self.CELL_SIZE * 0.2))
                    end = (center[0] + math.cos(angle) * (self.CELL_SIZE * 0.7), center[1] + math.sin(angle) * (self.CELL_SIZE * 0.7))
                    pygame.draw.line(self.screen, self.COLOR_PLAYER, start, end, 2)
            elif effect['type'] == 'flash':
                flash_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                flash_surf.fill(self.COLOR_FLASH)
                self.screen.blit(flash_surf, (0, 0))
        
        self.effects.clear()

    def _render_ui(self):
        score_text = f"DONUTS: {self.score} / {self.NUM_DONUTS}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))
        
        time_text = f"STEPS: {self.steps} / {self.TIME_LIMIT_STEPS}"
        time_surf = self.font_large.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, time_surf.get_rect(topright=(self.WIDTH - 10, 10)))
        
        if self.game_over:
            if not self.donuts: msg = "YOU WIN!"
            elif self.player_pos in self.obstacles: msg = "GAME OVER - OBSTACLE"
            else: msg = "GAME OVER - TIME OUT"
                
            end_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            end_rect = end_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            bg_rect = end_rect.inflate(20, 10)
            pygame.draw.rect(self.screen, self.COLOR_BG, bg_rect)
            pygame.draw.rect(self.screen, self.COLOR_GRID, bg_rect, 2)
            self.screen.blit(end_surf, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,) and self.action_space.nvec.tolist() == [5, 2, 2]
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3) and obs.dtype == np.uint8
        assert isinstance(info, dict) and "score" in info and "steps" in info
        action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3) and obs.dtype == np.uint8
        assert isinstance(reward, (int, float)) and isinstance(term, bool) and not trunc
        assert isinstance(info, dict) and "score" in info and "steps" in info
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    env.validate_implementation()
    
    obs, info = env.reset()
    done = False
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Donut Grid")
    clock = pygame.time.Clock()
    
    action = np.array([0, 0, 0])

    print("\n--- Human Controls ---")
    print(env.user_guide)
    print("Press Q to quit, R to reset.")
    
    while not done:
        human_action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                done = True
            if event.type == pygame.KEYDOWN:
                human_action_taken = True
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    human_action_taken = False
                else:
                    action[0] = 0
                    
        if human_action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action[0]}, Reward: {reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            action[0] = 0
            if terminated:
                print("--- Episode Finished ---")

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    env.close()