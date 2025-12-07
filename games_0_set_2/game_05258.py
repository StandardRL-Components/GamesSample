import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set Pygame to run in a headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your player (blue square). "
        "Push the brown crates onto the green targets before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game where you must race against the clock to push crates "
        "onto their designated targets. Plan your moves carefully to solve the puzzle in 60 seconds!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- CONSTANTS ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 16, 10
        self.GRID_SIZE = 40
        self.FPS = 30
        self.MAX_STEPS = self.FPS * 60  # 60-second timer

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_WALL = (80, 80, 90)
        self.COLOR_FLOOR = (40, 50, 60)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GLOW = (150, 200, 255)
        self.COLOR_CRATE = (200, 120, 50)
        self.COLOR_CRATE_ON_TARGET = (220, 180, 90)
        self.COLOR_TARGET = (50, 200, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_PARTICLE = (255, 255, 255)

        # --- GYMNASIUM SPACES ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- PYGAME SETUP ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 72)
        self.font_ui = pygame.font.Font(None, 32)

        # --- GAME STATE (initialized in reset) ---
        self.player_pos = None
        self.crate_pos = None
        self.target_pos = None
        self.walls = None
        self.timer = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.particles = None
        self.crates_on_target_history = None
        self.player_anim_timer = 0
        self.player_anim_dir = (0, 0)
        
        # This will be initialized in the first call to reset()
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.timer = 60.0
        self.particles = []
        self.crates_on_target_history = 0

        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        # 1. Create walls (a border and some internal obstacles)
        self.walls = set()
        for x in range(self.GRID_WIDTH):
            self.walls.add((x, 0))
            self.walls.add((x, self.GRID_HEIGHT - 1))
        for y in range(self.GRID_HEIGHT):
            self.walls.add((0, y))
            self.walls.add((self.GRID_WIDTH - 1, y))

        # Add some random internal walls
        for _ in range(self.np_random.integers(5, 10)):
            wall_x = self.np_random.integers(2, self.GRID_WIDTH - 2)
            wall_y = self.np_random.integers(2, self.GRID_HEIGHT - 2)
            self.walls.add((wall_x, wall_y))

        # 2. Get all valid floor tiles
        floor_tiles = []
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if (x, y) not in self.walls:
                    floor_tiles.append((x, y))

        self.np_random.shuffle(floor_tiles)

        # 3. Place targets
        self.target_pos = [tuple(pos) for pos in floor_tiles[:3]]

        # 4. Place crates on targets (solved state)
        self.crate_pos = [tuple(pos) for pos in self.target_pos]

        # 5. "Pull" crates randomly to create a solvable puzzle
        num_pulls = self.np_random.integers(15, 30)
        for _ in range(num_pulls):
            crate_idx = self.np_random.integers(0, len(self.crate_pos))
            crate_to_pull = self.crate_pos[crate_idx]

            possible_pulls = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_crate_pos = (crate_to_pull[0] + dx, crate_to_pull[1] + dy)
                player_pos_for_pull = (crate_to_pull[0] - dx, crate_to_pull[1] - dy)

                # Check if new crate pos and player pos are valid
                if (new_crate_pos not in self.walls and
                    new_crate_pos not in self.crate_pos and
                    player_pos_for_pull not in self.walls):
                    possible_pulls.append(new_crate_pos)

            if possible_pulls:
                # FIX: np.random.choice returns a numpy array. Convert it to a tuple
                # to ensure self.crate_pos remains a list of tuples, preventing the
                # ValueError on the 'in' check.
                chosen_pull = self.np_random.choice(possible_pulls)
                self.crate_pos[crate_idx] = tuple(chosen_pull)

        # 6. Place player
        occupied = set(self.crate_pos) | set(self.target_pos) | self.walls
        valid_player_spawns = [pos for pos in floor_tiles if pos not in occupied]
        
        # FIX: np.random.choice returns a numpy array. Convert it to a tuple.
        if valid_player_spawns:
            self.player_pos = tuple(self.np_random.choice(valid_player_spawns))
        else:
            self.player_pos = floor_tiles[-1]


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]

        # --- UPDATE GAME LOGIC ---
        self.steps += 1
        self.timer = max(0, self.timer - 1.0 / self.FPS)

        # Animate player squash
        if self.player_anim_timer > 0:
            self.player_anim_timer -= 1

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.2)

        # --- REWARD CALCULATION ---
        reward = -0.01  # Small penalty for time passing

        # --- ACTION HANDLING ---
        move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}  # up, down, left, right
        dx, dy = move_map.get(movement, (0, 0))

        if dx != 0 or dy != 0:
            self.player_anim_timer = 4  # frames
            self.player_anim_dir = (dx, dy)

            next_px, next_py = self.player_pos[0] + dx, self.player_pos[1] + dy

            # Case 1: Move into empty space or wall
            if (next_px, next_py) in self.walls:
                pass  # Blocked
            elif (next_px, next_py) not in self.crate_pos:
                self.player_pos = (next_px, next_py)

            # Case 2: Move into a crate (try to push)
            else:
                try:
                    crate_idx = self.crate_pos.index((next_px, next_py))
                    crate_to_push = self.crate_pos[crate_idx]

                    next_cx, next_cy = crate_to_push[0] + dx, crate_to_push[1] + dy

                    # Check if crate's destination is valid
                    if (next_cx, next_cy) not in self.walls and (next_cx, next_cy) not in self.crate_pos:
                        # Calculate reward for pushing closer/further from target
                        # Find the corresponding target. This assumes a fixed order, which might be brittle.
                        # A better approach would be to find the closest free target, but for now, we'll keep the logic.
                        target_for_crate = self.target_pos[crate_idx] 
                        dist_before = math.hypot(crate_to_push[0] - target_for_crate[0], crate_to_push[1] - target_for_crate[1])
                        dist_after = math.hypot(next_cx - target_for_crate[0], next_cy - target_for_crate[1])

                        if dist_after < dist_before:
                            reward += 0.1
                        else:
                            reward -= 0.2

                        # Move crate and player
                        self.crate_pos[crate_idx] = (next_cx, next_cy)
                        self.player_pos = (next_px, next_py)

                        # Spawn push particles
                        for _ in range(10):
                            vel_x = -dx * self.np_random.uniform(0.5, 2.0) + self.np_random.uniform(-0.5, 0.5)
                            vel_y = -dy * self.np_random.uniform(0.5, 2.0) + self.np_random.uniform(-0.5, 0.5)
                            self.particles.append({
                                'pos': [(next_px + 0.5) * self.GRID_SIZE, (next_py + 0.5) * self.GRID_SIZE],
                                'vel': [vel_x, vel_y],
                                'life': self.np_random.integers(10, 20),
                                'size': self.np_random.uniform(2, 5)
                            })
                except ValueError:
                    # This can happen if multiple crates are pushed into the same spot, which shouldn't be possible
                    # with current logic, but as a safeguard, we just treat it as a blocked move.
                    pass


        # Check for newly placed crates on targets
        crates_on_target_now = self._get_crates_on_target_count()
        newly_placed_crates = crates_on_target_now - self.crates_on_target_history
        if newly_placed_crates > 0:
            reward += newly_placed_crates * 1.0
        self.crates_on_target_history = crates_on_target_now

        # --- CHECK TERMINATION ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated or truncated:
            self.game_over = True
            if self.win:
                reward += 10.0
            elif self.timer <= 0:
                reward -= 10.0

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_crates_on_target_count(self):
        return sum(1 for crate in self.crate_pos if crate in self.target_pos)

    def _check_termination(self):
        if self._get_crates_on_target_count() == len(self.target_pos):
            self.win = True
            return True
        if self.timer <= 0:
            return True
        return False

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw floor
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if (x, y) not in self.walls:
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR, (x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE))

        # Draw targets
        for tx, ty in self.target_pos:
            pygame.draw.rect(self.screen, self.COLOR_TARGET, (tx * self.GRID_SIZE, ty * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE))

        # Draw walls
        for wx, wy in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, (wx * self.GRID_SIZE, wy * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE))

        # Draw crates
        for i, (cx, cy) in enumerate(self.crate_pos):
            color = self.COLOR_CRATE_ON_TARGET if (cx, cy) in self.target_pos else self.COLOR_CRATE
            crate_rect = pygame.Rect(cx * self.GRID_SIZE + 4, cy * self.GRID_SIZE + 4, self.GRID_SIZE - 8, self.GRID_SIZE - 8)
            pygame.draw.rect(self.screen, color, crate_rect, border_radius=4)

        # Draw player with squash/stretch animation
        px, py = self.player_pos
        player_rect = pygame.Rect(px * self.GRID_SIZE + 5, py * self.GRID_SIZE + 5, self.GRID_SIZE - 10, self.GRID_SIZE - 10)

        if self.player_anim_timer > 0:
            dx, dy = self.player_anim_dir
            squash_factor = (self.player_anim_timer / 4.0) * 4  # 4 pixels of squash
            if dx != 0:  # Horizontal movement
                player_rect.inflate_ip(-squash_factor, squash_factor)
            elif dy != 0:  # Vertical movement
                player_rect.inflate_ip(squash_factor, -squash_factor)

        # Player Glow
        glow_rect = player_rect.inflate(10, 10)
        shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, (*self.COLOR_PLAYER_GLOW, 80), shape_surf.get_rect(), border_radius=8)
        self.screen.blit(shape_surf, glow_rect)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=5)

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, self.COLOR_PARTICLE, (int(p['pos'][0]), int(p['pos'][1])), int(p['size']))

    def _render_ui(self):
        # Timer display
        timer_text = f"TIME: {self.timer:.1f}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 15, 10))

        # Crates on target display
        crates_text = f"CRATES: {self._get_crates_on_target_count()} / {len(self.target_pos)}"
        crates_surf = self.font_ui.render(crates_text, True, self.COLOR_TEXT)
        self.screen.blit(crates_surf, (self.SCREEN_WIDTH - crates_surf.get_width() - 15, 40))

        # Game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            message = "YOU WIN!" if self.win else "TIME'S UP!"
            color = self.COLOR_TARGET if self.win else self.COLOR_CRATE

            msg_surf = self.font_large.render(message, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "crates_on_target": self._get_crates_on_target_count(),
        }

    def close(self):
        pygame.quit()