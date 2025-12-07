import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:06:07.992330
# Source Brief: brief_00811.md
# Brief Index: 811
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Stack falling blocks to clear lines and score points before time runs out. "
        "Unstable placements will be penalized."
    )
    user_guide = (
        "Controls: ←→ to move the falling block. Press space to instantly drop it."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.FPS = 20  # As per mechanics brief
        self.MAX_STEPS = 2400  # 120 seconds * 20 FPS
        self.WIN_SCORE = 200
        self.LEVEL_UP_SCORE = 20
        self.BLOCK_SIZE = 20
        self.PLAYFIELD_WIDTH_IN_BLOCKS = 10
        self.PLAYFIELD_HEIGHT_IN_BLOCKS = 19
        self.PLAYFIELD_WIDTH = self.PLAYFIELD_WIDTH_IN_BLOCKS * self.BLOCK_SIZE
        self.PLAYFIELD_HEIGHT = self.PLAYFIELD_HEIGHT_IN_BLOCKS * self.BLOCK_SIZE
        self.PLAYFIELD_X = (self.SCREEN_WIDTH - self.PLAYFIELD_WIDTH) // 2
        self.PLAYFIELD_Y = (self.SCREEN_HEIGHT - self.PLAYFIELD_HEIGHT) // 2 + 5
        self.PLAYER_MOVE_SPEED = 1 # As per brief
        self.INITIAL_FALL_SPEED = 1.0
        self.FALL_SPEED_INCREMENT = 0.05
        
        # --- Colors & Fonts ---
        self.COLOR_BG = (20, 25, 35)
        self.COLOR_BOUNDARY = (200, 200, 220)
        self.COLOR_GRID = (30, 35, 45)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_FLASH = (255, 255, 255)
        self.BLOCK_COLORS = [
            (231, 76, 60),  # Red
            (46, 204, 113), # Green
            (52, 152, 219), # Blue
            (241, 196, 15),  # Yellow
            (155, 89, 182),  # Magenta
        ]
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 24)

        # Game state variables are initialized in reset()
        self.placed_blocks = []
        self.current_block = None
        self.current_block_pos = [0,0]
        self.fall_speed = 0.0
        self.score = 0
        self.level = 0
        self.steps = 0
        self.game_over = False
        self.line_clear_flash = []
        self.particles = []
        
        # Initialize state variables
        self.reset()
        # self.validate_implementation() # Removed for final submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.placed_blocks = []
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.score = 0
        self.level = 1
        self.steps = 0
        self.game_over = False
        self.line_clear_flash = []
        self.particles = []
        
        self._spawn_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle player input ---
        if self.current_block:
            # Horizontal Movement
            if movement == 3: # Left
                self.current_block_pos[0] -= self.PLAYER_MOVE_SPEED
            elif movement == 4: # Right
                self.current_block_pos[0] += self.PLAYER_MOVE_SPEED
            
            # Clamp to playfield
            self.current_block_pos[0] = max(self.PLAYFIELD_X, self.current_block_pos[0])
            self.current_block_pos[0] = min(self.PLAYFIELD_X + self.PLAYFIELD_WIDTH - self.BLOCK_SIZE, self.current_block_pos[0])

            # Instant Drop
            if space_held:
                # Sound: Fast drop whoosh
                final_y = self._get_drop_y()
                self.current_block_pos[1] = final_y
                reward += self._handle_landing()
            else:
                # Normal Gravity
                self.current_block_pos[1] += self.fall_speed
                if self._check_collision():
                    self.current_block_pos[1] = round((self.current_block_pos[1] - self.PLAYFIELD_Y) / self.BLOCK_SIZE) * self.BLOCK_SIZE + self.PLAYFIELD_Y
                    reward += self._handle_landing()
        
        self._update_particles()

        # --- Check for termination conditions ---
        terminated = self.game_over
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            reward -= 100
            terminated = True
        
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

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
            "level": self.level,
            "time_remaining": (self.MAX_STEPS - self.steps) / self.FPS
        }

    def _spawn_block(self):
        x = self.PLAYFIELD_X + (self.PLAYFIELD_WIDTH_IN_BLOCKS // 2 - 1) * self.BLOCK_SIZE
        y = self.PLAYFIELD_Y
        color = random.choice(self.BLOCK_COLORS)
        self.current_block = {"rect": pygame.Rect(x, y, self.BLOCK_SIZE, self.BLOCK_SIZE), "color": color}
        self.current_block_pos = [float(x), float(y)]

        if self._check_collision():
            # Sound: Game over failure
            self.game_over = True
            self.current_block = None

    def _check_collision(self):
        if not self.current_block: return False
        
        block_rect = pygame.Rect(int(self.current_block_pos[0]), int(self.current_block_pos[1]), self.BLOCK_SIZE, self.BLOCK_SIZE)

        # Check floor collision
        if block_rect.bottom > self.PLAYFIELD_Y + self.PLAYFIELD_HEIGHT:
            return True
        
        # Check collision with placed blocks
        for placed in self.placed_blocks:
            if block_rect.colliderect(placed["rect"]):
                return True
        return False
    
    def _get_drop_y(self):
        original_y = self.current_block_pos[1]
        test_y = original_y
        while True:
            test_y += 1
            block_rect = pygame.Rect(int(self.current_block_pos[0]), int(test_y), self.BLOCK_SIZE, self.BLOCK_SIZE)
            if block_rect.bottom > self.PLAYFIELD_Y + self.PLAYFIELD_HEIGHT:
                return test_y - 1
            
            collided = False
            for placed in self.placed_blocks:
                if block_rect.colliderect(placed["rect"]):
                    collided = True
                    break
            if collided:
                return test_y - 1
    
    def _handle_landing(self):
        if not self.current_block: return 0
        # Sound: Block place thud
        
        final_rect = pygame.Rect(int(self.current_block_pos[0]), int(self.current_block_pos[1]), self.BLOCK_SIZE, self.BLOCK_SIZE)
        
        # Snap to grid
        final_rect.x = round((final_rect.x - self.PLAYFIELD_X) / self.BLOCK_SIZE) * self.BLOCK_SIZE + self.PLAYFIELD_X
        final_rect.y = round((final_rect.y - self.PLAYFIELD_Y) / self.BLOCK_SIZE) * self.BLOCK_SIZE + self.PLAYFIELD_Y

        new_block = {"rect": final_rect, "color": self.current_block["color"]}
        self.placed_blocks.append(new_block)
        
        reward = 0.1 # Reward for placing a block
        
        # Check for unstable placement
        support_blocks = [p for p in self.placed_blocks if p["rect"].top == final_rect.bottom and final_rect.colliderect(pygame.Rect(p["rect"].x, p["rect"].y-1, p["rect"].width, p["rect"].height))]
        if not support_blocks and final_rect.bottom < self.PLAYFIELD_Y + self.PLAYFIELD_HEIGHT:
             # Hanging in mid-air, very unstable
             reward -= 5
        elif support_blocks:
            support_min_x = min(s["rect"].left for s in support_blocks)
            support_max_x = max(s["rect"].right for s in support_blocks)
            if final_rect.left < support_min_x - 2 or final_rect.right > support_max_x + 2:
                reward -= 5 # Unstable overhang
        
        self.current_block = None
        reward += self._check_line_clears()
        self._spawn_block()
        return reward
        
    def _check_line_clears(self):
        lines_cleared = 0
        reward = 0
        
        y_coords = sorted(list(set(p["rect"].y for p in self.placed_blocks)), reverse=True)
        
        for y in y_coords:
            blocks_in_row = [p for p in self.placed_blocks if p["rect"].y == y]
            if len(blocks_in_row) == self.PLAYFIELD_WIDTH_IN_BLOCKS:
                # Sound: Line clear success
                lines_cleared += 1
                reward += 10
                self.line_clear_flash.append({"y": y, "timer": 5})
                self._create_particles_for_line(y)

                # Remove cleared line
                self.placed_blocks = [p for p in self.placed_blocks if p["rect"].y != y]
                
                # Shift blocks above down
                for p in self.placed_blocks:
                    if p["rect"].y < y:
                        p["rect"].y += self.BLOCK_SIZE
        
        if lines_cleared > 0:
            new_score = self.score + lines_cleared * 10
            if (new_score // self.LEVEL_UP_SCORE) > (self.score // self.LEVEL_UP_SCORE):
                # Sound: Level up chime
                self.level += 1
                self.fall_speed += self.FALL_SPEED_INCREMENT
            self.score = new_score

        return reward
    
    def _create_particles_for_line(self, y):
        for _ in range(30):
            x = self.PLAYFIELD_X + random.uniform(0, self.PLAYFIELD_WIDTH)
            pos = [x, y + self.BLOCK_SIZE / 2]
            vel = [random.uniform(-1, 1), random.uniform(-2, 0)]
            life = random.randint(10, 20)
            color = random.choice([self.COLOR_FLASH, self.COLOR_BOUNDARY])
            size = random.uniform(1, 4)
            self.particles.append({"pos": pos, "vel": vel, "life": life, "color": color, "size": size})

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]
        
    def _render_game(self):
        # Draw playfield boundary
        pygame.draw.rect(self.screen, self.COLOR_BOUNDARY, (self.PLAYFIELD_X-2, self.PLAYFIELD_Y-2, self.PLAYFIELD_WIDTH+4, self.PLAYFIELD_HEIGHT+4), 2, border_radius=3)
        
        # Draw grid
        for i in range(1, self.PLAYFIELD_WIDTH_IN_BLOCKS):
            x = self.PLAYFIELD_X + i * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.PLAYFIELD_Y), (x, self.PLAYFIELD_Y + self.PLAYFIELD_HEIGHT))
        for i in range(1, self.PLAYFIELD_HEIGHT_IN_BLOCKS):
            y = self.PLAYFIELD_Y + i * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.PLAYFIELD_X, y), (self.PLAYFIELD_X + self.PLAYFIELD_WIDTH, y))

        # Draw placed blocks
        for block in self.placed_blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            pygame.draw.rect(self.screen, self.COLOR_BG, block["rect"], 1)

        # Draw current block
        if self.current_block:
            # Draw ghost block
            ghost_y = self._get_drop_y()
            ghost_rect = pygame.Rect(int(self.current_block_pos[0]), int(ghost_y), self.BLOCK_SIZE, self.BLOCK_SIZE)
            ghost_rect.x = round((ghost_rect.x - self.PLAYFIELD_X) / self.BLOCK_SIZE) * self.BLOCK_SIZE + self.PLAYFIELD_X
            ghost_color = self.current_block["color"][:3] + (50,) # Add alpha
            s = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(s, ghost_color, (0,0, self.BLOCK_SIZE, self.BLOCK_SIZE), border_radius=2)
            self.screen.blit(s, (ghost_rect.x, ghost_rect.y))
            
            # Draw actual block
            current_rect = pygame.Rect(int(self.current_block_pos[0]), int(self.current_block_pos[1]), self.BLOCK_SIZE, self.BLOCK_SIZE)
            pygame.draw.rect(self.screen, self.current_block["color"], current_rect)
            pygame.draw.rect(self.screen, self.COLOR_FLASH, current_rect, 2)
            
        # Draw line clear flash
        new_flashes = []
        for flash in self.line_clear_flash:
            alpha = int(255 * (flash["timer"] / 5))
            flash_surface = pygame.Surface((self.PLAYFIELD_WIDTH, self.BLOCK_SIZE), pygame.SRCALPHA)
            flash_surface.fill((self.COLOR_FLASH[0], self.COLOR_FLASH[1], self.COLOR_FLASH[2], alpha))
            self.screen.blit(flash_surface, (self.PLAYFIELD_X, flash["y"]))
            flash["timer"] -= 1
            if flash["timer"] > 0:
                new_flashes.append(flash)
        self.line_clear_flash = new_flashes
        
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1])), int(p["size"]))

    def _render_ui(self):
        # Score Display
        score_text = self.font_large.render(f"SCORE", True, self.COLOR_TEXT)
        score_val = self.font_large.render(f"{self.score:04d}", True, self.COLOR_FLASH)
        self.screen.blit(score_text, (20, 20))
        self.screen.blit(score_val, (20, 55))
        
        # Time Display
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = self.font_large.render(f"TIME", True, self.COLOR_TEXT)
        time_val = self.font_large.render(f"{time_left:05.1f}", True, self.COLOR_FLASH)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_val.get_width() - 20, 20))
        self.screen.blit(time_val, (self.SCREEN_WIDTH - time_val.get_width() - 20, 55))
        
        # Level Display
        level_text = self.font_small.render(f"LEVEL {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH // 2 - level_text.get_width() // 2, 20))

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
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Manual Play ---
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a separate display for rendering if not using "human" mode
    pygame.display.set_caption("Block Stacker")
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    total_reward = 0
    
    while not done:
        # Action mapping for human control
        movement = 0 # None
        space = 0 # Released
        shift = 0 # Released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Render the observation to the display
        render_surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(render_surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS) # Control the speed of the game
        
    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    pygame.quit()