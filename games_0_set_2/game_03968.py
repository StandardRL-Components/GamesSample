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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your character (red square) "
        "and push the blue blocks into the green goals."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game. Push all the blocks to their matching goals "
        "before the timer runs out to win. Plan your moves carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    GRID_WIDTH = 20
    GRID_HEIGHT = 12
    TILE_SIZE = 32

    PLAY_AREA_WIDTH = GRID_WIDTH * TILE_SIZE
    PLAY_AREA_HEIGHT = GRID_HEIGHT * TILE_SIZE

    OFFSET_X = (SCREEN_WIDTH - PLAY_AREA_WIDTH) // 2
    OFFSET_Y = (SCREEN_HEIGHT - PLAY_AREA_HEIGHT) // 2

    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 45)
    COLOR_PLAYER = (255, 50, 50)
    COLOR_BLOCK = (50, 100, 255)
    COLOR_GOAL = (50, 255, 100)
    COLOR_GOAL_ACTIVE = (200, 255, 220)
    COLOR_TEXT = (240, 240, 240)

    NUM_BLOCKS = 3
    MAX_STEPS = 900  # 30 seconds at 30 FPS
    PLAYER_MOVE_COOLDOWN = 6  # frames

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 20, bold=True)

        # Initialize state variables
        self.player_pos = (0, 0)
        self.blocks = []
        self.goals = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_move_timer = 0
        self.particles = []
        self.blocks_on_goals_indices = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_move_timer = 0
        self.particles = []
        self.blocks_on_goals_indices = set()

        self._generate_level()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def _generate_level(self):
        """Generates a new random level layout."""
        all_positions = set((x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT))

        # Place goals
        goal_positions_array = self.np_random.choice(list(all_positions), size=self.NUM_BLOCKS, replace=False)
        self.goals = [tuple(pos) for pos in goal_positions_array]
        for pos in self.goals:
            all_positions.remove(pos)

        # Place blocks near goals
        self.blocks = []
        for goal_pos in self.goals:
            possible_block_pos = []
            for pos in all_positions:
                dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
                if 2 <= dist <= 5:
                    possible_block_pos.append(pos)

            if not possible_block_pos:  # Fallback if no ideal spot is found
                possible_block_pos = list(all_positions)

            block_pos = tuple(self.np_random.choice(possible_block_pos, size=1)[0])
            self.blocks.append(block_pos)
            all_positions.remove(block_pos)

        # Place player
        self.player_pos = tuple(self.np_random.choice(list(all_positions), size=1)[0])

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = 0.0

        # Store state for reward calculation
        old_block_positions = list(self.blocks)

        # Update game logic
        self._handle_input(action)
        self._update_particles()

        # Check for newly scored blocks
        newly_scored_indices = self._update_scored_blocks()
        if newly_scored_indices:
            reward += len(newly_scored_indices) * 1.0  # +1 per block scored
            for idx in newly_scored_indices:
                # Sound effect placeholder: # sfx_score.play()
                self._create_particles(self.blocks[idx])

        # Calculate distance-based rewards
        for i in range(self.NUM_BLOCKS):
            old_dist = abs(old_block_positions[i][0] - self.goals[i][0]) + abs(old_block_positions[i][1] - self.goals[i][1])
            new_dist = abs(self.blocks[i][0] - self.goals[i][0]) + abs(self.blocks[i][1] - self.goals[i][1])
            if new_dist < old_dist:
                reward += 0.1
            elif new_dist > old_dist:
                reward -= 0.01

        self.steps += 1
        terminated = self._check_termination()

        # Terminal rewards
        if terminated:
            if len(self.blocks_on_goals_indices) == self.NUM_BLOCKS:
                reward += 100.0  # Win bonus
                # Sound effect placeholder: # sfx_win.play()
            else:
                reward -= 50.0  # Timeout penalty
                # Sound effect placeholder: # sfx_lose.play()

        self.score += reward

        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, False, info

    def _handle_input(self, action):
        if self.game_over:
            return

        movement = action[0]

        if self.player_move_timer > 0:
            self.player_move_timer -= 1
            return

        if movement == 0:  # No-op
            return

        self.player_move_timer = self.PLAYER_MOVE_COOLDOWN

        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1  # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1  # Right

        target_x, target_y = self.player_pos[0] + dx, self.player_pos[1] + dy

        # Check for block at target position
        if (target_x, target_y) in self.blocks:
            block_idx = self.blocks.index((target_x, target_y))
            block_target_x, block_target_y = target_x + dx, target_y + dy

            # Check if block can be pushed
            is_wall = not (0 <= block_target_x < self.GRID_WIDTH and 0 <= block_target_y < self.GRID_HEIGHT)
            is_other_block = (block_target_x, block_target_y) in self.blocks

            if not is_wall and not is_other_block:
                # Push block and move player
                self.blocks[block_idx] = (block_target_x, block_target_y)
                self.player_pos = (target_x, target_y)
                # Sound effect placeholder: # sfx_push.play()
        # Check for empty space
        elif (target_x, target_y) not in self.blocks:
            # Check for wall collision
            if 0 <= target_x < self.GRID_WIDTH and 0 <= target_y < self.GRID_HEIGHT:
                self.player_pos = (target_x, target_y)
                # Sound effect placeholder: # sfx_move.play()

    def _update_scored_blocks(self):
        """Checks which blocks are on goals and returns newly scored indices."""
        current_on_goals = set()
        for i, block_pos in enumerate(self.blocks):
            # The comparison is now between two tuples, which is valid.
            if block_pos == self.goals[i]:
                current_on_goals.add(i)

        newly_scored = current_on_goals - self.blocks_on_goals_indices
        self.blocks_on_goals_indices = current_on_goals
        return newly_scored

    def _check_termination(self):
        win_condition = len(self.blocks_on_goals_indices) == self.NUM_BLOCKS
        lose_condition = self.steps >= self.MAX_STEPS

        if win_condition or lose_condition:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(self.OFFSET_X + x * self.TILE_SIZE,
                                   self.OFFSET_Y + y * self.TILE_SIZE,
                                   self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw goals
        for i, pos in enumerate(self.goals):
            rect = pygame.Rect(self.OFFSET_X + pos[0] * self.TILE_SIZE,
                               self.OFFSET_Y + pos[1] * self.TILE_SIZE,
                               self.TILE_SIZE, self.TILE_SIZE)
            color = self.COLOR_GOAL_ACTIVE if i in self.blocks_on_goals_indices else self.COLOR_GOAL
            pygame.draw.rect(self.screen, color, rect.inflate(-4, -4))

        # Draw blocks
        for i, pos in enumerate(self.blocks):
            rect = pygame.Rect(self.OFFSET_X + pos[0] * self.TILE_SIZE,
                               self.OFFSET_Y + pos[1] * self.TILE_SIZE,
                               self.TILE_SIZE, self.TILE_SIZE)

            # Draw a darker border for depth
            pygame.draw.rect(self.screen, tuple(c * 0.7 for c in self.COLOR_BLOCK), rect)
            pygame.draw.rect(self.screen, self.COLOR_BLOCK, rect.inflate(-4, -4))

        # Draw player
        player_rect = pygame.Rect(self.OFFSET_X + self.player_pos[0] * self.TILE_SIZE,
                                  self.OFFSET_Y + self.player_pos[1] * self.TILE_SIZE,
                                  self.TILE_SIZE, self.TILE_SIZE)

        # Glow effect
        glow_center = player_rect.center
        glow_radius = int(self.TILE_SIZE * 0.8)
        glow_color = (*self.COLOR_PLAYER, 100)  # RGBA
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (glow_center[0] - glow_radius, glow_center[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Player square
        pygame.draw.rect(self.screen, tuple(c * 0.7 for c in self.COLOR_PLAYER), player_rect)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-4, -4))

        # Draw particles
        self._render_particles()

    def _render_ui(self):
        # Time remaining
        time_left = max(0, (self.MAX_STEPS - self.steps) / 30.0)
        time_text = self.font_small.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (20, 15))

        # Blocks scored
        blocks_text = self.font_small.render(f"GOALS: {len(self.blocks_on_goals_indices)}/{self.NUM_BLOCKS}", True, self.COLOR_TEXT)
        text_rect = blocks_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 15))
        self.screen.blit(blocks_text, text_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if len(self.blocks_on_goals_indices) == self.NUM_BLOCKS:
                end_text_str = "LEVEL CLEAR!"
                end_text_color = self.COLOR_GOAL
            else:
                end_text_str = "TIME UP!"
                end_text_color = self.COLOR_PLAYER

            end_text = self.font_large.render(end_text_str, True, end_text_color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _create_particles(self, grid_pos):
        px, py = self._grid_to_pixel(grid_pos)
        px += self.TILE_SIZE // 2
        py += self.TILE_SIZE // 2
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            size = self.np_random.integers(2, 5)
            self.particles.append({"pos": [px, py], "vel": vel, "life": life, "max_life": life, "size": size})

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95  # friction
            p["vel"][1] *= 0.95
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p["life"] / p["max_life"]
            color = tuple(c * life_ratio for c in self.COLOR_GOAL_ACTIVE)
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            size = int(p["size"] * life_ratio)
            if size > 0:
                pygame.draw.rect(self.screen, color, (pos[0] - size // 2, pos[1] - size // 2, size, size))

    def _grid_to_pixel(self, grid_pos):
        return (self.OFFSET_X + grid_pos[0] * self.TILE_SIZE,
                self.OFFSET_Y + grid_pos[1] * self.TILE_SIZE)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "blocks_in_goal": len(self.blocks_on_goals_indices),
            "max_steps": self.MAX_STEPS
        }

    def close(self):
        pygame.quit()