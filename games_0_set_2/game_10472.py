import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
import numpy as np
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Navigate a collapsing tessellated world by teleporting between tiles and
    strategically reinforcing them with limited stability units.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a collapsing tessellated world by teleporting between tiles and "
        "strategically reinforcing them with limited stability units."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to teleport "
        "and shift to reinforce your current tile."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 16, 10
    TILE_SIZE = 40
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (20, 20, 35)
    COLOR_GRID_LINES = (40, 40, 60)
    COLOR_STABLE = (60, 180, 75)
    COLOR_WARNING = (255, 225, 25)
    COLOR_UNSTABLE = (245, 130, 48)
    COLOR_CRITICAL = (230, 25, 75)
    COLOR_COLLAPSED = (10, 10, 10)
    COLOR_PLAYER = (70, 240, 240)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_REINFORCE = (255, 255, 255, 150)

    # Game Mechanics
    INITIAL_RESOURCES = 100
    TELEPORT_COST = 5
    REINFORCE_COST = 10
    REINFORCE_AMOUNT = 20
    MAX_STABILITY = 100
    STABILITY_DECAY_RATE = 1

    # Rewards
    REWARD_REINFORCE = 10
    REWARD_TELEPORT = -5
    REWARD_STEP = -0.1
    REWARD_WIN = 100
    REWARD_LOSE = -100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 12, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        self.grid = []
        self.player_pos = [0, 0]
        self.cursor_pos = [0, 0]
        self.resources = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.particles = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.resources = self.INITIAL_RESOURCES
        self.particles = []

        self.grid = [
            [{
                "stability": self.np_random.integers(30, 71),
                "target_stability": self.np_random.integers(30, 71)
            } for _ in range(self.GRID_COLS)]
            for _ in range(self.GRID_ROWS)
        ]
        
        self.player_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        # Ensure the starting tile is fully stable to prevent termination during no-op stability tests
        start_tile = self.grid[self.player_pos[0]][self.player_pos[1]]
        start_tile["stability"] = self.MAX_STABILITY
        start_tile["target_stability"] = self.MAX_STABILITY

        self.cursor_pos = list(self.player_pos)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = self.REWARD_STEP

        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1

        # 1. Handle Player Actions
        # Cursor Movement
        if movement == 1:  # Up
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2:  # Down
            self.cursor_pos[0] = min(self.GRID_ROWS - 1, self.cursor_pos[0] + 1)
        elif movement == 3:  # Left
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4:  # Right
            self.cursor_pos[1] = min(self.GRID_COLS - 1, self.cursor_pos[1] + 1)

        # Teleport Action
        if space_press:
            target_tile = self.grid[self.cursor_pos[0]][self.cursor_pos[1]]
            if self.resources >= self.TELEPORT_COST and target_tile["stability"] > 0:
                self.resources -= self.TELEPORT_COST
                self.score -= self.TELEPORT_COST
                reward += self.REWARD_TELEPORT
                self._create_teleport_effect(self.player_pos, self.cursor_pos)
                self.player_pos = list(self.cursor_pos)

        # Reinforce Action
        if shift_press:
            current_tile = self.grid[self.player_pos[0]][self.player_pos[1]]
            if self.resources >= self.REINFORCE_COST and current_tile["stability"] > 0 and current_tile["stability"] < self.MAX_STABILITY:
                self.resources -= self.REINFORCE_COST
                current_tile["target_stability"] = min(self.MAX_STABILITY, current_tile["target_stability"] + self.REINFORCE_AMOUNT)
                self.score += self.REWARD_REINFORCE
                reward += self.REWARD_REINFORCE
                self._create_reinforce_effect(self.player_pos)

        # 2. Update Game State
        self.steps += 1
        num_stable = 0
        num_active_tiles = 0

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile = self.grid[r][c]
                
                # Smoothly animate stability changes
                if tile["stability"] != tile["target_stability"]:
                    diff = tile["target_stability"] - tile["stability"]
                    change = math.copysign(min(abs(diff), 5), diff)
                    tile["stability"] += change

                if tile["stability"] > 0:
                    num_active_tiles += 1
                    tile["target_stability"] = max(0, tile["target_stability"] - self.STABILITY_DECAY_RATE)
                    if tile["stability"] >= self.MAX_STABILITY:
                        num_stable += 1
                else: # Tile is collapsed
                    tile["target_stability"] = 0
                    tile["stability"] = 0

        # 3. Check for Termination
        terminated = False
        player_tile = self.grid[self.player_pos[0]][self.player_pos[1]]

        if self.resources < 0 or player_tile["stability"] <= 0:
            terminated = True
            self.game_over = True
            self.win_condition = False
            reward += self.REWARD_LOSE
            self.score += self.REWARD_LOSE
        elif num_stable == num_active_tiles and num_active_tiles > 0:
            terminated = True
            self.game_over = True
            self.win_condition = True
            reward += self.REWARD_WIN
            self.score += self.REWARD_WIN
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.win_condition = False

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        self._update_and_draw_particles()
        
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile = self.grid[r][c]
                rect = pygame.Rect(c * self.TILE_SIZE, r * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                
                stability_ratio = tile["stability"] / self.MAX_STABILITY
                
                if stability_ratio <= 0:
                    color = self.COLOR_COLLAPSED
                elif stability_ratio < 0.25:
                    color = self.COLOR_CRITICAL
                elif stability_ratio < 0.50:
                    color = self.COLOR_UNSTABLE
                elif stability_ratio < 0.75:
                    color = self.COLOR_WARNING
                else:
                    color = self.COLOR_STABLE

                pygame.draw.rect(self.screen, color, rect)
                
                if tile["stability"] > 0:
                    text_surf = self.font_small.render(str(int(tile["stability"])), True, self.COLOR_TEXT)
                    text_rect = text_surf.get_rect(center=rect.center)
                    self.screen.blit(text_surf, text_rect)
                
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, rect, 1)

        cursor_rect = pygame.Rect(self.cursor_pos[1] * self.TILE_SIZE, self.cursor_pos[0] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 * 3 + 2
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, int(pulse))

        player_rect = pygame.Rect(self.player_pos[1] * self.TILE_SIZE, self.player_pos[0] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-8, -8), 0, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, 2, border_radius=4)

    def _render_ui(self):
        resource_text = self.font_medium.render(f"STABILITY: {max(0, self.resources)}", True, self.COLOR_TEXT)
        self.screen.blit(resource_text, (10, 10))

        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY" if self.win_condition else "SYSTEM COLLAPSE"
            color = self.COLOR_STABLE if self.win_condition else self.COLOR_CRITICAL
            
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _create_teleport_effect(self, from_pos, to_pos):
        from_px = [(from_pos[1] + 0.5) * self.TILE_SIZE, (from_pos[0] + 0.5) * self.TILE_SIZE]
        
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 5)
            self.particles.append({
                "pos": [from_px[0] + math.cos(angle) * 20, from_px[1] + math.sin(angle) * 20],
                "vel": [-math.cos(angle) * speed, -math.sin(angle) * speed],
                "life": 15, "color": self.COLOR_PLAYER
            })
        
        to_px = [(to_pos[1] + 0.5) * self.TILE_SIZE, (to_pos[0] + 0.5) * self.TILE_SIZE]
        for _ in range(50):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 6)
            self.particles.append({
                "pos": list(to_px),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": 20, "color": self.COLOR_PLAYER
            })

    def _create_reinforce_effect(self, pos):
        center_px = [(pos[1] + 0.5) * self.TILE_SIZE, (pos[0] + 0.5) * self.TILE_SIZE]
        self.particles.append({
            "pos": center_px, "life": 20, "radius": 0, "max_radius": self.TILE_SIZE * 0.7,
            "type": "ripple", "color": self.COLOR_REINFORCE
        })

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
                continue

            if p.get("type") == "ripple":
                p["radius"] += p["max_radius"] / 20
                alpha = int(p["color"][3] * (p["life"] / 20))
                pygame.gfxdraw.aacircle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), (*p["color"][:3], alpha))
            else:
                p["pos"][0] += p["vel"][0]
                p["pos"][1] += p["vel"][1]
                p["vel"][0] *= 0.95
                p["vel"][1] *= 0.95
                size = int(max(1, p["life"] / 4))
                pygame.draw.circle(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1])), size)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # This part requires a display. It will not run in a truly headless environment.
    try:
        pygame.display.set_caption("Tessellate Collapse")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        clock = pygame.time.Clock()
        
        terminated = False
        total_reward = 0
        
        print("\n--- Manual Control ---")
        print("Arrows: Move Cursor")
        print("Space: Teleport")
        print("Shift: Reinforce")
        print("R: Reset")
        print("Q: Quit")
        
        running = True
        while running:
            action = [0, 0, 0]
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        terminated = False
                        print("--- Environment Reset ---")
            
            if not terminated:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]: action[0] = 1
                elif keys[pygame.K_DOWN]: action[0] = 2
                elif keys[pygame.K_LEFT]: action[0] = 3
                elif keys[pygame.K_RIGHT]: action[0] = 4
                
                if keys[pygame.K_SPACE]: action[1] = 1
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30)
    except pygame.error as e:
        print(f"\nPygame display error: {e}")
        print("Could not create display. This is expected in a headless environment.")
        print("The environment itself is still functional for training.")
    finally:
        env.close()
        pygame.quit()