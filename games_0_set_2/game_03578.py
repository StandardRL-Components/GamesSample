
# Generated: 2025-08-27T23:47:11.820958
# Source Brief: brief_03578.md
# Brief Index: 3578

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: Use arrow keys to move. Collect all the gems before you run out of moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Collect all the gems on the grid within the move limit to maximize your score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 10
        self.CELL_SIZE = 36
        self.GEM_COUNT = 25
        self.MOVE_LIMIT = 50

        # Calculate grid offsets for centering
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT) // 2

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_UI_BG = (30, 30, 50, 180) # RGBA
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 220, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Etc...        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = 0
        self.player_pos = [0, 0]
        self.gems = []
        self.last_gem_pos = None
        self.particles = []
        self.np_random = None
        self.win_message = ""
        
        # Initialize state variables
        self.reset()
        
        # Self-validation
        # self.validate_implementation() # Commented out for final submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.MOVE_LIMIT
        self.win_message = ""
        
        # Generate unique positions for player and gems
        all_positions = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
        self.np_random.shuffle(all_positions)

        player_start_pos = all_positions.pop()
        self.player_pos = list(player_start_pos)

        gem_positions = all_positions[:self.GEM_COUNT]
        self.gems = []
        for pos in gem_positions:
            self.gems.append({
                "pos": list(pos),
                "color": self.np_random.choice(self.GEM_COLORS, axis=0).tolist()
            })
        
        self.last_gem_pos = None
        self.particles = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Update game logic
        self.steps += 1
        reward = 0
        
        prev_pos = list(self.player_pos)

        if movement == 1:  # Up
            if self.player_pos[1] > 0: self.player_pos[1] -= 1
        elif movement == 2:  # Down
            if self.player_pos[1] < self.GRID_SIZE - 1: self.player_pos[1] += 1
        elif movement == 3:  # Left
            if self.player_pos[0] > 0: self.player_pos[0] -= 1
        elif movement == 4:  # Right
            if self.player_pos[0] < self.GRID_SIZE - 1: self.player_pos[0] += 1
        # movement == 0 is a no-op, but still consumes a move

        self.moves_remaining -= 1

        # Check for gem collection
        gem_collected = False
        for i, gem in enumerate(self.gems):
            if self.player_pos == gem["pos"]:
                gem_collected = True
                collected_gem = self.gems.pop(i)
                # sfx: gem collect sound
                
                # Base reward and score
                reward = 1.0
                self.score += 10

                # Bonus reward for distance
                if self.last_gem_pos:
                    dist = abs(prev_pos[0] - self.last_gem_pos[0]) + abs(prev_pos[1] - self.last_gem_pos[1])
                    if dist > 3:
                        reward += 5.0
                        self.score += 50
                
                self.last_gem_pos = list(self.player_pos)
                
                # Spawn particles for visual feedback
                pixel_pos = self._grid_to_pixel(self.player_pos)
                self._spawn_particles(pixel_pos, collected_gem["color"], 30)
                break
        
        if not gem_collected:
            reward = -0.2

        self._update_particles()
        
        # Check termination conditions
        terminated = self._check_termination(reward)
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _check_termination(self, current_reward):
        if len(self.gems) == 0:
            current_reward += 50.0
            self.score += 500
            self.game_over = True
            self.win_message = "YOU WIN!"
            # sfx: win fanfare
            return True
        elif self.moves_remaining <= 0:
            current_reward -= 50.0
            self.game_over = True
            self.win_message = "OUT OF MOVES"
            # sfx: lose sound
            return True
        return False
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_grid()
        self._render_gems()
        self._render_particles()
        self._render_player()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "gems_remaining": len(self.gems),
        }

    # --- Helper and Rendering Methods ---

    def _grid_to_pixel(self, grid_pos):
        px = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return [px, py]

    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal lines
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_WIDTH, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _render_gems(self):
        gem_radius = self.CELL_SIZE // 3
        for gem in self.gems:
            px, py = self._grid_to_pixel(gem["pos"])
            color = gem["color"]
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), gem_radius, color)
            pygame.gfxdraw.aacircle(self.screen, int(px), int(py), gem_radius, color)
            highlight_radius = gem_radius // 3
            pygame.gfxdraw.filled_circle(self.screen, int(px - highlight_radius), int(py - highlight_radius), highlight_radius, (255, 255, 255, 150))

    def _render_player(self):
        player_size = self.CELL_SIZE // 2
        px, py = self._grid_to_pixel(self.player_pos)
        player_rect = pygame.Rect(px - player_size // 2, py - player_size // 2, player_size, player_size)

        glow_size = int(player_size * 1.8)
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (255, 255, 255, 50), glow_surf.get_rect(), border_radius=5)
        self.screen.blit(glow_surf, (player_rect.centerx - glow_size // 2, player_rect.centery - glow_size // 2))

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = random.uniform(2, 5)
            lifetime = random.randint(20, 40)
            self.particles.append({"pos": list(pos), "vel": vel, "radius": radius, "color": color, "life": lifetime})

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # Gravity
            p["life"] -= 1
            p["radius"] -= 0.05
        self.particles = [p for p in self.particles if p["life"] > 0 and p["radius"] > 0]

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            radius = int(p["radius"])
            if radius > 0:
                alpha = int(255 * (p["life"] / 40))
                color = (*p["color"], max(0, min(255, alpha)))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

    def _render_ui(self):
        ui_bar = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_bar.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_bar, (0, 0))

        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 8))

        moves_str = f"Moves: {self.moves_remaining}"
        moves_color = (255, 100, 100) if self.moves_remaining <= 10 else self.COLOR_TEXT
        moves_text = self.font_main.render(moves_str, True, moves_color)
        moves_text_rect = moves_text.get_rect(right=self.SCREEN_WIDTH - 15, top=8)
        self.screen.blit(moves_text, moves_text_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("✓ Running implementation validation...")
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
        assert self.moves_remaining == self.MOVE_LIMIT
        assert len(self.gems) == self.GEM_COUNT

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        assert self.moves_remaining == self.MOVE_LIMIT - 1
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment for interactive play
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True

    print("\n" + "="*30)
    print("Gem Collector - Interactive Test")
    print("="*30)
    print(env.user_guide)
    print("Press 'R' to reset.")
    print("Press ESC or close the window to quit.")
    print("="*30 + "\n")

    while running:
        movement_action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    print("\n--- Game Reset ---")
                
                if not env.game_over:
                    if event.key == pygame.K_UP: movement_action = 1
                    elif event.key == pygame.K_DOWN: movement_action = 2
                    elif event.key == pygame.K_LEFT: movement_action = 3
                    elif event.key == pygame.K_RIGHT: movement_action = 4
                    
                    if movement_action != 0:
                        action = np.array([movement_action, 0, 0])
                        obs, reward, terminated, truncated, info = env.step(action)
                        print(f"Action: {action[0]}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_remaining']}")
                        if terminated:
                            print(f"--- Episode Finished ---")
                            print(f"Final Score: {info['score']}")

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    env.close()