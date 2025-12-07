import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Stack falling blocks of varying sizes. Create vertical chains of 5 or more same-sized blocks to clear them and score points before time runs out."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to move the falling block. Press ↓ or space to drop the block faster."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLAY_AREA_WIDTH = 320
    PLAY_AREA_HEIGHT = 360
    CELL_SIZE = 40
    GRID_WIDTH = PLAY_AREA_WIDTH // CELL_SIZE
    GRID_HEIGHT = PLAY_AREA_HEIGHT // CELL_SIZE
    FPS = 60
    GAME_DURATION_SECONDS = 90
    WIN_SCORE = 300
    CHAIN_REQUIREMENT = 5
    MULTIPLIER_COOLDOWN_SECONDS = 5

    # --- Colors ---
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (30, 40, 60)
    COLOR_BORDER = (80, 90, 110)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 15)

    BLOCK_TYPES = {
        1: {'color': (0, 200, 255), 'name': 'small'},  # Cyan
        2: {'color': (255, 0, 150), 'name': 'medium'}, # Magenta
        3: {'color': (255, 220, 0), 'name': 'large'},  # Yellow
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20, bold=True)

        self.play_area_x_start = (self.SCREEN_WIDTH - self.PLAY_AREA_WIDTH) // 2
        self.play_area_y_start = self.SCREEN_HEIGHT - self.PLAY_AREA_HEIGHT
        self.play_area_rect = pygame.Rect(
            self.play_area_x_start, self.play_area_y_start,
            self.PLAY_AREA_WIDTH, self.PLAY_AREA_HEIGHT
        )

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid = []
        self.stacked_blocks = {}
        self.current_block = None
        self.time_limit = self.GAME_DURATION_SECONDS * self.FPS
        self.base_fall_speed = 1.0
        self.current_fall_speed = 0.0
        self.score_multiplier = 1
        self.multiplier_timer = 0
        self.particles = []
        self.clearing_blocks = []
        self.next_block_id = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.grid = [[None for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        self.stacked_blocks = {}
        self.next_block_id = 0

        self.base_fall_speed = 1.0
        self.current_fall_speed = self.base_fall_speed

        self.score_multiplier = 1
        self.multiplier_timer = 0

        self.particles = []
        self.clearing_blocks = []

        self._spawn_new_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        self._handle_input(action)
        reward += self._update_physics()
        self._update_animations()

        self.current_fall_speed = self.base_fall_speed + (self.steps // (30 * self.FPS)) * 0.5

        if self.multiplier_timer > 0:
            self.multiplier_timer -= 1
            if self.multiplier_timer == 0:
                self.score_multiplier = 1
                # sfx: multiplier_reset

        terminated = self._check_termination()

        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100 # Goal-oriented reward
            else:
                reward -= 10 # Penalty for timeout

        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if self.current_block:
            # Horizontal Movement
            if movement == 3: # Left
                self.current_block['x'] = max(self.play_area_x_start, self.current_block['x'] - self.CELL_SIZE)
            elif movement == 4: # Right
                max_x = self.play_area_x_start + self.PLAY_AREA_WIDTH - self.current_block['size'] * self.CELL_SIZE
                self.current_block['x'] = min(max_x, self.current_block['x'] + self.CELL_SIZE)

            # Accelerated Drop
            if movement == 2 or space_held:
                self.current_block['y'] += self.CELL_SIZE / 2 # Fast drop
                # sfx: fast_drop

    def _update_physics(self):
        reward = 0
        if not self.current_block:
            return reward

        self.current_block['y'] += self.current_fall_speed

        grid_x = int((self.current_block['x'] - self.play_area_x_start) / self.CELL_SIZE)
        grid_y = int((self.current_block['y'] - self.play_area_y_start) / self.CELL_SIZE)

        # Check for collision
        landed = False
        if self.current_block['y'] + self.CELL_SIZE >= self.SCREEN_HEIGHT:
            landed = True
            self.current_block['y'] = self.SCREEN_HEIGHT - self.CELL_SIZE
        else:
            for i in range(self.current_block['size']):
                check_x, check_y = grid_x + i, grid_y + 1
                if 0 <= check_x < self.GRID_WIDTH and 0 <= check_y < self.GRID_HEIGHT:
                    if self.grid[check_y][check_x] is not None:
                        landed = True
                        self.current_block['y'] = self.play_area_y_start + (check_y - 1) * self.CELL_SIZE
                        break

        if landed:
            reward += self._place_block()
            chain_reward = self._check_and_process_chains()
            if chain_reward > 0:
                reward += chain_reward
                self._settle_stack()
            self._spawn_new_block()

            if self.current_block is None: # Game over on spawn
                self.game_over = True

        return reward

    def _place_block(self):
        # sfx: place_block
        block_id = self.next_block_id
        self.next_block_id += 1

        grid_x = int((self.current_block['x'] - self.play_area_x_start) / self.CELL_SIZE)
        grid_y = int((self.current_block['y'] - self.play_area_y_start) / self.CELL_SIZE)

        # Clamp to grid
        grid_y = min(grid_y, self.GRID_HEIGHT - 1)

        self.stacked_blocks[block_id] = {
            'id': block_id,
            'size': self.current_block['size'],
            'color': self.current_block['color'],
            'grid_x': grid_x,
            'grid_y': grid_y
        }

        for i in range(self.current_block['size']):
            if 0 <= grid_x + i < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT:
                self.grid[grid_y][grid_x + i] = block_id

        self.current_block = None
        return 0.1 # Reward for placing a block

    def _spawn_new_block(self):
        size = self.np_random.choice(list(self.BLOCK_TYPES.keys()))
        color = self.BLOCK_TYPES[size]['color']

        start_grid_x = self.np_random.integers(0, self.GRID_WIDTH - size + 1)
        start_x = self.play_area_x_start + start_grid_x * self.CELL_SIZE

        # Check if spawn area is blocked
        if self.grid[0][start_grid_x] is not None:
            self.current_block = None # Game over condition
            return

        self.current_block = {
            'x': start_x,
            'y': self.play_area_y_start - self.CELL_SIZE,
            'size': size,
            'color': color
        }

    def _check_and_process_chains(self):
        blocks_to_clear = set()
        for x in range(self.GRID_WIDTH):
            current_run = []
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                block_id = self.grid[y][x]
                if block_id is not None and block_id in self.stacked_blocks:
                    block = self.stacked_blocks[block_id]
                    if not current_run or block['size'] == current_run[0]['size']:
                        current_run.append(block)
                    else:
                        if len(current_run) >= self.CHAIN_REQUIREMENT:
                            for b in current_run:
                                blocks_to_clear.add(b['id'])
                        current_run = [block]
                else:
                    if len(current_run) >= self.CHAIN_REQUIREMENT:
                        for b in current_run:
                            blocks_to_clear.add(b['id'])
                    current_run = []
            if len(current_run) >= self.CHAIN_REQUIREMENT:
                for b in current_run:
                    blocks_to_clear.add(b['id'])

        if not blocks_to_clear:
            return 0

        # sfx: clear_chain
        num_cleared = 0
        for block_id in blocks_to_clear:
            if block_id in self.stacked_blocks:
                block = self.stacked_blocks[block_id]
                self.clearing_blocks.append({**block, 'timer': 15}) # 0.25s animation
                self._create_particles(block)
                num_cleared += 1
                del self.stacked_blocks[block_id]

        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x] in blocks_to_clear:
                    self.grid[y][x] = None

        reward = num_cleared * self.score_multiplier
        self.score += int(num_cleared * 10 * self.score_multiplier)
        self.score_multiplier += 1
        self.multiplier_timer = self.MULTIPLIER_COOLDOWN_SECONDS * self.FPS
        return reward

    def _settle_stack(self):
        moved = True
        while moved:
            moved = False
            sorted_blocks = sorted(self.stacked_blocks.values(), key=lambda b: b['grid_y'], reverse=True)
            for block in sorted_blocks:
                can_fall = True
                new_y = block['grid_y']

                # Check if block still exists (it might have been part of another chain)
                if block['id'] not in self.stacked_blocks:
                    continue

                while can_fall:
                    fall_to_y = new_y + 1
                    if fall_to_y >= self.GRID_HEIGHT:
                        can_fall = False
                        break

                    for i in range(block['size']):
                        if self.grid[fall_to_y][block['grid_x'] + i] is not None:
                            can_fall = False
                            break
                    if can_fall:
                        new_y = fall_to_y

                if new_y != block['grid_y']:
                    moved = True
                    old_y = block['grid_y']
                    # Clear old grid position
                    for i in range(block['size']):
                        self.grid[old_y][block['grid_x'] + i] = None
                    # Set new grid position
                    block['grid_y'] = new_y
                    for i in range(block['size']):
                        self.grid[new_y][block['grid_x'] + i] = block['id']

        # Check for new chains after settling
        chain_reward = self._check_and_process_chains()
        if chain_reward > 0:
            self._settle_stack() # Recursive call for cascade reactions

    def _update_animations(self):
        # Update clearing blocks
        self.clearing_blocks = [b for b in self.clearing_blocks if b['timer'] > 0]
        for block in self.clearing_blocks:
            block['timer'] -= 1

        # Update particles
        self.particles = [p for p in self.particles if p['alpha'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['alpha'] -= 5
            p['size'] = max(0, p['size'] - 0.1)

    def _create_particles(self, block):
        cx = self.play_area_x_start + block['grid_x'] * self.CELL_SIZE + (block['size'] * self.CELL_SIZE / 2)
        cy = self.play_area_y_start + block['grid_y'] * self.CELL_SIZE + self.CELL_SIZE / 2
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': [cx, cy],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'size': random.uniform(2, 6),
                'color': block['color'],
                'alpha': 255
            })

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            return True
        if self.steps >= self.time_limit:
            return True
        if self.current_block is None and not self.game_over:
             # This happens if a block can't spawn
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            px = self.play_area_x_start + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.play_area_y_start), (px, self.SCREEN_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.play_area_y_start + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.play_area_x_start, py), (self.play_area_x_start + self.PLAY_AREA_WIDTH, py))

        # Draw play area border
        pygame.draw.rect(self.screen, self.COLOR_BORDER, self.play_area_rect, 3)

        # Draw stacked blocks
        for block in self.stacked_blocks.values():
            rect = pygame.Rect(
                self.play_area_x_start + block['grid_x'] * self.CELL_SIZE,
                self.play_area_y_start + block['grid_y'] * self.CELL_SIZE,
                block['size'] * self.CELL_SIZE,
                self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, block['color'], rect)
            pygame.draw.rect(self.screen, self.COLOR_BG, rect, 2)

        # Draw clearing blocks (flashing)
        for block in self.clearing_blocks:
            if block['timer'] % 4 < 2:
                rect = pygame.Rect(
                    self.play_area_x_start + block['grid_x'] * self.CELL_SIZE,
                    self.play_area_y_start + block['grid_y'] * self.CELL_SIZE,
                    block['size'] * self.CELL_SIZE,
                    self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, (255, 255, 255), rect)

        # Draw current block
        if self.current_block:
            rect = pygame.Rect(
                int(self.current_block['x']), int(self.current_block['y']),
                self.current_block['size'] * self.CELL_SIZE, self.CELL_SIZE
            )
            # Glow effect
            glow_rect = rect.inflate(10, 10)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*self.current_block['color'], 50), glow_surf.get_rect(), border_radius=8)
            self.screen.blit(glow_surf, glow_rect.topleft)

            pygame.draw.rect(self.screen, self.current_block['color'], rect)
            pygame.draw.rect(self.screen, self.COLOR_BG, rect, 2)

        # Draw particles
        for p in self.particles:
            color = (*p['color'], int(p['alpha']))
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'][0]), int(p['pos'][1]),
                int(p['size']), color
            )

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score}"
        self._draw_text(score_text, (20, 10), self.font_large)

        # Time
        time_left = max(0, (self.time_limit - self.steps) / self.FPS)
        time_text = f"TIME: {time_left:.1f}"
        self._draw_text(time_text, (self.SCREEN_WIDTH - 20, 10), self.font_large, align="right")

        # Multiplier
        if self.score_multiplier > 1:
            multi_text = f"x{self.score_multiplier}"
            color = tuple(min(255, c + (self.score_multiplier-1)*20) for c in (255, 100, 0))
            self._draw_text(multi_text, (20, 50), self.font_small, color=color)

        if self.game_over:
            end_text = "VICTORY!" if self.score >= self.WIN_SCORE else "TIME UP!"
            self._draw_text(end_text, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 50), self.font_large, align="center")

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, align="left"):
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_surf = font.render(text, True, color)

        if align == "right":
            text_rect = text_surf.get_rect(topright=pos)
        elif align == "center":
            text_rect = text_surf.get_rect(center=pos)
        else: # left
            text_rect = text_surf.get_rect(topleft=pos)

        shadow_rect = text_rect.move(2, 2)

        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "multiplier": self.score_multiplier,
            "time_left": (self.time_limit - self.steps) / self.FPS
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually.
    # It will not be executed by the test suite.
    
    # Un-comment the line below to run with a visible display
    os.environ.pop("SDL_VIDEODRIVER", None)

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Human Player Controls ---
    # Arrows: Move
    # Space: Fast Drop
    # Q: Quit
    
    # Mapping keys to MultiDiscrete actions
    key_map = {
        pygame.K_LEFT:  [3, 0, 0],
        pygame.K_RIGHT: [4, 0, 0],
        pygame.K_DOWN:  [2, 0, 0],
        pygame.K_SPACE: [0, 1, 0],
    }

    pygame.display.set_caption("Chain Reaction Stacker")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    total_reward = 0
    while not done:
        action = [0, 0, 0] # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                done = True
        
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        if keys[pygame.K_SPACE]:
            action[1] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()