
# Generated: 2025-08-28T03:50:17.375225
# Source Brief: brief_05058.md
# Brief Index: 5058

        
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

    user_guide = (
        "Controls: Use arrow keys to move the selection cursor. Press space to select a tile. "
        "Find all 15 hidden shapes before the time runs out!"
    )

    game_description = (
        "A timed hidden object game. Scan the isometric grid to find all 15 hidden shapes. "
        "Objects become harder to see as time ticks down. Find them all to win!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 18, 18
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 20, 10
    ORIGIN_X, ORIGIN_Y = SCREEN_WIDTH // 2, 60
    
    NUM_OBJECTS = 15
    MAX_STEPS = 1800  # 60 seconds at 30 FPS

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TIMER_BAR = (0, 200, 100)
    COLOR_TIMER_BAR_BG = (70, 80, 100)
    
    OBJECT_PALETTE = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255),
        (255, 255, 80), (255, 80, 255), (80, 255, 255),
        (255, 150, 50), (50, 150, 255)
    ]
    OBJECT_SHAPES = ['circle', 'square', 'triangle', 'diamond']

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_small = pygame.font.SysFont("Consolas", 18)
            self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
            self.font_huge = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 22)
            self.font_large = pygame.font.Font(None, 36)
            self.font_huge = pygame.font.Font(None, 52)
            
        self.np_random = None
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.last_space_held = False
        
        self._generate_objects()
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _generate_objects(self):
        self.objects = []
        occupied_positions = set()
        
        while len(self.objects) < self.NUM_OBJECTS:
            pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
            if pos not in occupied_positions:
                obj = {
                    "pos": pos,
                    "color": random.choice(self.OBJECT_PALETTE),
                    "shape": random.choice(self.OBJECT_SHAPES),
                    "found": False,
                }
                self.objects.append(obj)
                occupied_positions.add(pos)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        self._handle_movement(movement)
        
        space_press = space_held and not self.last_space_held
        if space_press:
            selection_reward = self._handle_selection()
            reward += selection_reward
        self.last_space_held = space_held
        
        hovering_object = self._get_object_at_cursor()
        if hovering_object and not hovering_object["found"]:
            reward += 0.1
            
        self.steps += 1
        self._update_particles()
        
        terminated = self._check_termination()
        if terminated:
            if self.win_state:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Timeout penalty
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_movement(self, movement):
        if movement == 1:  # Up
            self.cursor_pos[1] -= 1
        elif movement == 2:  # Down
            self.cursor_pos[1] += 1
        elif movement == 3:  # Left
            self.cursor_pos[0] -= 1
        elif movement == 4:  # Right
            self.cursor_pos[0] += 1
            
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

    def _get_object_at_cursor(self):
        for obj in self.objects:
            if obj["pos"][0] == self.cursor_pos[0] and obj["pos"][1] == self.cursor_pos[1]:
                return obj
        return None

    def _handle_selection(self):
        obj = self._get_object_at_cursor()
        if obj and not obj["found"]:
            obj["found"] = True
            # SFX: Correct find sound
            center_pos = self._iso_to_screen(self.cursor_pos[0] + 0.5, self.cursor_pos[1] + 0.5)
            self._create_particles(center_pos, obj["color"])
            return 10.0
        # SFX: Incorrect selection sound
        return -1.0

    def _check_termination(self):
        found_count = sum(1 for obj in self.objects if obj["found"])
        if found_count == self.NUM_OBJECTS:
            self.game_over = True
            self.win_state = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win_state = False
            return True
        return False

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
            "objects_found": sum(1 for obj in self.objects if obj["found"]),
            "cursor_pos": self.cursor_pos,
        }

    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_WIDTH + 1):
            p1 = self._iso_to_screen(i, 0)
            p2 = self._iso_to_screen(i, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
        for i in range(self.GRID_HEIGHT + 1):
            p1 = self._iso_to_screen(0, i)
            p2 = self._iso_to_screen(self.GRID_WIDTH, i)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)

        # Draw objects
        visibility = max(0.1, 1.0 - (self.steps / self.MAX_STEPS) * 0.9)
        alpha = int(255 * visibility)
        
        for obj in self.objects:
            if not obj["found"]:
                self._draw_iso_shape(obj["shape"], obj["color"], obj["pos"], alpha)

        # Draw particles
        self._draw_particles()

        # Draw cursor
        cx, cy = self.cursor_pos
        points = [
            self._iso_to_screen(cx, cy),
            self._iso_to_screen(cx + 1, cy),
            self._iso_to_screen(cx + 1, cy + 1),
            self._iso_to_screen(cx, cy + 1),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CURSOR)
        pygame.draw.polygon(self.screen, self.COLOR_CURSOR, points, 2)

    def _draw_iso_shape(self, shape, color, pos, alpha):
        center_pos = self._iso_to_screen(pos[0] + 0.5, pos[1] + 0.5)
        
        temp_surface = pygame.Surface((self.TILE_WIDTH_HALF * 2, self.TILE_WIDTH_HALF * 2), pygame.SRCALPHA)
        temp_surface.set_colorkey((0,0,0))
        
        local_center = (self.TILE_WIDTH_HALF, self.TILE_WIDTH_HALF)
        
        if shape == 'circle':
            radius = int(self.TILE_WIDTH_HALF * 0.4)
            pygame.gfxdraw.filled_circle(temp_surface, local_center[0], local_center[1], radius, color)
            pygame.gfxdraw.aacircle(temp_surface, local_center[0], local_center[1], radius, color)
        elif shape == 'square':
            size = int(self.TILE_WIDTH_HALF * 0.8)
            rect = pygame.Rect(0, 0, size, size)
            rect.center = local_center
            pygame.draw.rect(temp_surface, color, rect)
        elif shape == 'triangle':
            size = int(self.TILE_WIDTH_HALF * 0.5)
            points = [
                (local_center[0], local_center[1] - size),
                (local_center[0] - size, local_center[1] + size),
                (local_center[0] + size, local_center[1] + size),
            ]
            pygame.gfxdraw.filled_polygon(temp_surface, points, color)
            pygame.gfxdraw.aapolygon(temp_surface, points, color)
        elif shape == 'diamond':
            size = int(self.TILE_WIDTH_HALF * 0.5)
            points = [
                (local_center[0], local_center[1] - size),
                (local_center[0] - size, local_center[1]),
                (local_center[0], local_center[1] + size),
                (local_center[0] + size, local_center[1]),
            ]
            pygame.gfxdraw.filled_polygon(temp_surface, points, color)
            pygame.gfxdraw.aapolygon(temp_surface, points, color)

        temp_surface.set_alpha(alpha)
        self.screen.blit(temp_surface, (center_pos[0] - local_center[0], center_pos[1] - local_center[1]))

    def _render_ui(self):
        # Timer bar
        timer_width = self.SCREEN_WIDTH - 20
        time_left_ratio = max(0, 1 - (self.steps / self.MAX_STEPS))
        current_width = int(timer_width * time_left_ratio)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR_BG, (10, 10, timer_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, (10, 10, current_width, 20))
        
        # Found count
        found_count = sum(1 for obj in self.objects if obj["found"])
        found_text = f"FOUND: {found_count} / {self.NUM_OBJECTS}"
        text_surf = self.font_large.render(found_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 20, 35))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        text_surf = self.font_large.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (20, 35))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win_state else "TIME'S UP!"
            color = (100, 255, 100) if self.win_state else (255, 100, 100)
            
            msg_surf = self.font_huge.render(message, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(msg_surf, msg_rect)

            final_score_surf = self.font_large.render(f"Final Score: {int(self.score)}", True, self.COLOR_TEXT)
            final_score_rect = final_score_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 40))
            self.screen.blit(final_score_surf, final_score_rect)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[2] > 0]
        for p in self.particles:
            p[0][0] += p[1][0] # pos_x += vel_x
            p[0][1] += p[1][1] # pos_y += vel_y
            p[1][1] += 0.2     # gravity
            p[2] -= 1          # life
            p[3] = max(0, p[3] - 4) # alpha fade

    def _create_particles(self, pos, color):
        for _ in range(20):
            vel = [self.np_random.uniform(-2.5, 2.5), self.np_random.uniform(-4, -1)]
            life = self.np_random.integers(20, 40)
            self.particles.append([[pos[0], pos[1]], vel, life, 255])

    def _draw_particles(self):
        for p in self.particles:
            pos, _, life, alpha = p
            radius = int(max(1, life * 0.15))
            color = (self.COLOR_CURSOR[0], self.COLOR_CURSOR[1], self.COLOR_CURSOR[2], alpha)
            
            # Use a temporary surface for alpha blending
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (radius, radius), radius)
            self.screen.blit(temp_surf, (int(pos[0]) - radius, int(pos[1]) - radius))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        assert info['objects_found'] == 0
        assert len(self.objects) == self.NUM_OBJECTS
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a different window for rendering
    pygame.display.set_caption("Hidden Object Game")
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    done = False
    total_reward = 0
    
    # Game loop
    while not done:
        # --- Human Controls ---
        movement = 0 # no-op
        space = 0
        shift = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Rendering ---
        # Convert the observation back to a Pygame surface
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Frame Rate ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(30)

    print(f"Game Over! Final Score: {info['score']:.2f}")
    
    # Keep the final screen visible for a few seconds
    pygame.time.wait(3000)
    
    env.close()