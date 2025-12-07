import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Guard:
    def __init__(self, path, cell_size, ui_margin):
        self.path = path
        self.cell_size = cell_size
        self.ui_margin = ui_margin
        self.path_index = 0
        self.grid_pos = self.path[0]
        self.pixel_pos = pygame.math.Vector2(
            self.grid_pos[0] * self.cell_size + self.cell_size / 2,
            self.grid_pos[1] * self.cell_size + self.cell_size / 2 + self.ui_margin
        )
        self.speed = 1.0
        self.direction = pygame.math.Vector2(0, -1)
        self.vision_range = 5.5 * self.cell_size
        self.vision_angle = 45  # degrees

    def update(self):
        target_grid_pos = self.path[self.path_index]
        target_pixel_pos = pygame.math.Vector2(
            target_grid_pos[0] * self.cell_size + self.cell_size / 2,
            target_grid_pos[1] * self.cell_size + self.cell_size / 2 + self.ui_margin
        )

        vec_to_target = target_pixel_pos - self.pixel_pos
        if vec_to_target.length() < self.speed:
            self.pixel_pos = target_pixel_pos
            self.path_index = (self.path_index + 1) % len(self.path)
        else:
            if vec_to_target.length() > 0:
                self.direction = vec_to_target.normalize()
            self.pixel_pos += self.direction * self.speed
        
        self.grid_pos = (
            int(self.pixel_pos.x // self.cell_size),
            int((self.pixel_pos.y - self.ui_margin) // self.cell_size)
        )

    def is_player_in_cone(self, player_pixel_pos):
        vec_to_player = player_pixel_pos - self.pixel_pos
        dist_to_player = vec_to_player.length()

        if dist_to_player > self.vision_range or dist_to_player == 0:
            return False

        angle_to_player = self.direction.angle_to(vec_to_player)
        return abs(angle_to_player) < self.vision_angle / 2

    def draw(self, surface, steps):
        # Vision Cone
        cone_color = (*self.COLOR_VISION, 10 + int(10 * math.sin(steps * 0.2)))
        p1 = self.pixel_pos
        
        dir_left = self.direction.rotate(-self.vision_angle / 2) * self.vision_range
        p2 = self.pixel_pos + dir_left
        
        dir_right = self.direction.rotate(self.vision_angle / 2) * self.vision_range
        p3 = self.pixel_pos + dir_right
        
        pygame.gfxdraw.filled_trigon(surface, int(p1.x), int(p1.y), int(p2.x), int(p2.y), int(p3.x), int(p3.y), cone_color)
        pygame.gfxdraw.aatrigon(surface, int(p1.x), int(p1.y), int(p2.x), int(p2.y), int(p3.x), int(p3.y), cone_color)
        
        # Guard Body
        pos = (int(self.pixel_pos.x), int(self.pixel_pos.y))
        pygame.draw.circle(surface, self.COLOR_GUARD_GLOW, pos, 8)
        pygame.draw.circle(surface, self.COLOR_GUARD, pos, 6)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Avoid the red guards' vision cones. "
        "Collect blue power cells and reach the white escape ship to win."
    )

    game_description = (
        "A top-down stealth game. Sneak past patrolling alien guards to collect "
        "power cells and escape to your ship within the time limit."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GAME_DURATION_SECONDS = 60
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS
        
        self.CELL_SIZE = 32
        self.GRID_W = self.WIDTH // self.CELL_SIZE
        self.GRID_H = 12
        self.GAME_AREA_H = self.GRID_H * self.CELL_SIZE
        self.UI_MARGIN = (self.HEIGHT - self.GAME_AREA_H) // 2
        
        # Colors
        self.COLOR_BG = (10, 20, 35)
        self.COLOR_WALL = (40, 60, 80)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 60)
        self.COLOR_GUARD = (255, 50, 50)
        Guard.COLOR_GUARD = self.COLOR_GUARD
        self.COLOR_GUARD_GLOW = (255, 50, 50, 100)
        Guard.COLOR_GUARD_GLOW = self.COLOR_GUARD_GLOW
        self.COLOR_VISION = (255, 0, 0)
        Guard.COLOR_VISION = self.COLOR_VISION
        self.COLOR_CELL = (0, 150, 255)
        self.COLOR_CELL_GLOW = (0, 150, 255, 80)
        self.COLOR_SHIP = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_PATH = (80, 20, 20)

        # Fonts
        self.font_ui = pygame.font.Font(None, 36)
        self.font_msg = pygame.font.Font(None, 72)
        
        # Game State (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_grid_pos = (0,0)
        self.player_speed = 3.5
        self.walls = []
        self.power_cells = []
        self.cells_collected = 0
        self.ship_pos = (0, 0)
        self.guards = []
        self.time_remaining = 0
        
    def _create_level(self):
        self.walls = []
        wall_map = [
            "####################",
            "#                  #",
            "# ## ##### ## ##### #",
            "# ##         ##    #",
            "#    ### ###    ## #",
            "# ## ### ### ## ## #",
            "# ##         ##    #",
            "# ## ######### ## ##",
            "#    ### ###    ## #",
            "# ## ### ### ## ## #",
            "#                  #",
            "####################",
        ]
        for r, row_str in enumerate(wall_map):
            for c, char in enumerate(row_str):
                if char == '#':
                    self.walls.append(pygame.Rect(c * self.CELL_SIZE, r * self.CELL_SIZE + self.UI_MARGIN, self.CELL_SIZE, self.CELL_SIZE))

    def _get_valid_spawn(self):
        while True:
            x = self.np_random.integers(1, self.GRID_W - 1)
            y = self.np_random.integers(1, self.GRID_H - 1)
            rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE + self.UI_MARGIN, self.CELL_SIZE, self.CELL_SIZE)
            if not any(rect.colliderect(wall) for wall in self.walls):
                return (x, y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.time_remaining = self.GAME_DURATION_SECONDS
        
        self._create_level()
        
        # FIX: Moved player start position to a safer spot to prevent immediate detection.
        player_start_grid = (2, self.GRID_H - 2)
        self.player_pos = pygame.math.Vector2(
            player_start_grid[0] * self.CELL_SIZE + self.CELL_SIZE / 2,
            player_start_grid[1] * self.CELL_SIZE + self.CELL_SIZE / 2 + self.UI_MARGIN
        )
        self.player_grid_pos = player_start_grid

        self.power_cells = []
        self.cells_collected = 0
        for _ in range(3):
            pos = self._get_valid_spawn()
            self.power_cells.append(pygame.Rect(pos[0] * self.CELL_SIZE, pos[1] * self.CELL_SIZE + self.UI_MARGIN, self.CELL_SIZE, self.CELL_SIZE))

        self.ship_pos = (self.GRID_W // 2, 1)
        
        # Guard patrols
        path1 = [(2, 2), (8, 2), (8, 4), (2, 4)]
        path2 = [(17, 9), (11, 9), (11, 7), (17, 7)]
        path3 = [(5, 6), (5, 9), (8, 9), (8, 6)]
        self.guards = [
            Guard(path, self.CELL_SIZE, self.UI_MARGIN) for path in [path1, path2, path3]
        ]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        terminated = self.game_over
        truncated = False
        reward = 0.0

        if terminated:
            return self._get_observation(), reward, terminated, truncated, self._get_info()

        # --- If game is running ---
        self.steps += 1
        self.time_remaining -= 1 / self.FPS
        reward = 0.1  # Survival reward

        # Unpack action and update player
        movement = action[0]
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            next_pos = self.player_pos + pygame.math.Vector2(dx, dy) * self.player_speed
            player_rect = pygame.Rect(next_pos.x - 5, next_pos.y - 5, 10, 10)
            if not any(player_rect.colliderect(wall) for wall in self.walls):
                self.player_pos = next_pos
        
        self.player_grid_pos = (
            int(self.player_pos.x // self.CELL_SIZE),
            int((self.player_pos.y - self.UI_MARGIN) // self.CELL_SIZE)
        )

        # Update guards
        for guard in self.guards:
            guard.update()

        # Check interactions
        player_rect = pygame.Rect(self.player_pos.x - 5, self.player_pos.y - 5, 10, 10)
        
        # Power cell collection
        for cell in self.power_cells[:]:
            if player_rect.colliderect(cell):
                self.power_cells.remove(cell)
                self.cells_collected += 1
                self.score += 10
                reward += 10

        # Check for terminal states and truncation
        # Loss: Guard detection
        for guard in self.guards:
            if guard.is_player_in_cone(self.player_pos):
                self.game_over = True
                self.win_message = "DETECTED"
                reward -= 100
                self.score -= 100
                break
        
        # Loss: Time's up
        if not self.game_over and self.time_remaining <= 0:
            self.game_over = True
            self.win_message = "TIME'S UP"
            reward -= 100
            self.score -= 100

        # Win: Escaped
        if not self.game_over and self.player_grid_pos == self.ship_pos and self.cells_collected > 0:
            self.game_over = True
            self.win_message = "ESCAPED!"
            reward += 100
            self.score += 100

        # Non-terminal penalty: Guard proximity
        if not self.game_over:
            for guard in self.guards:
                dist = abs(self.player_grid_pos[0] - guard.grid_pos[0]) + abs(self.player_grid_pos[1] - guard.grid_pos[1])
                if dist <= 1:
                    reward -= 1
        
        terminated = self.game_over
        
        # Truncation: Max steps reached
        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _render_game(self):
        # Draw patrol paths
        for guard in self.guards:
            for i in range(len(guard.path)):
                start_grid = guard.path[i]
                end_grid = guard.path[(i + 1) % len(guard.path)]
                start_px = (start_grid[0] * self.CELL_SIZE + self.CELL_SIZE/2, start_grid[1] * self.CELL_SIZE + self.CELL_SIZE/2 + self.UI_MARGIN)
                end_px = (end_grid[0] * self.CELL_SIZE + self.CELL_SIZE/2, end_grid[1] * self.CELL_SIZE + self.CELL_SIZE/2 + self.UI_MARGIN)
                pygame.draw.line(self.screen, self.COLOR_PATH, start_px, end_px, 2)

        # Draw walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)
            
        # Draw escape ship
        blink_alpha = 150 + 105 * math.sin(self.steps * 0.1)
        ship_rect = pygame.Rect(self.ship_pos[0] * self.CELL_SIZE, self.ship_pos[1] * self.CELL_SIZE + self.UI_MARGIN, self.CELL_SIZE, self.CELL_SIZE)
        ship_color_surface = pygame.Surface(ship_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(ship_color_surface, (*self.COLOR_SHIP, blink_alpha), ship_color_surface.get_rect().inflate(-8, -8), 0, 4)
        self.screen.blit(ship_color_surface, ship_rect)

        # Draw power cells
        for cell in self.power_cells:
            glow_size = 18 + 6 * math.sin(self.steps * 0.2)
            pygame.draw.circle(self.screen, self.COLOR_CELL_GLOW, cell.center, int(glow_size))
            pygame.draw.rect(self.screen, self.COLOR_CELL, cell.inflate(-18, -18))

        # Draw guards
        for guard in self.guards:
            guard.draw(self.screen, self.steps)

        # Draw player
        player_int_pos = (int(self.player_pos.x), int(self.player_pos.y))
        pygame.draw.circle(self.screen, self.COLOR_PLAYER_GLOW, player_int_pos, 12)
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, player_int_pos, 7)

    def _render_ui(self):
        # Power Cell Count
        cell_text = self.font_ui.render(f"Cells: {self.cells_collected}", True, self.COLOR_TEXT)
        self.screen.blit(cell_text, (10, 10))

        # Timer
        time_str = f"Time: {max(0, int(self.time_remaining))}"
        time_text = self.font_ui.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            msg_text = self.font_msg.render(self.win_message, True, self.COLOR_SHIP)
            msg_rect = msg_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            # Use a separate surface for transparency
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            pygame.draw.rect(overlay, (0,0,0,150), msg_rect.inflate(20,20))
            self.screen.blit(overlay, (0,0))
            self.screen.blit(msg_text, msg_rect)

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
            "cells_collected": self.cells_collected,
            "time_remaining": self.time_remaining,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Stealth Escape")

    terminated = False
    truncated = False
    
    print(env.user_guide)

    while not terminated and not truncated:
        action = [0, 0, 0] # Default to no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        if not terminated:
            keys = pygame.key.get_pressed()
            
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
                
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)

        pygame.display.flip()
        env.clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}")
    pygame.time.wait(2000)
    
    env.close()