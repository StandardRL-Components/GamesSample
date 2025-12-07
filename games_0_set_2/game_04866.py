
# Generated: 2025-08-28T03:15:52.386673
# Source Brief: brief_04866.md
# Brief Index: 4866

        
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
    """
    A fast-paced, time-based puzzle game inspired by Sokoban.
    The player must push all crates onto their designated target locations
    before the 30-second timer runs out. The environment is designed for
    visual appeal and responsive gameplay, with smooth animations and
    clear feedback for all actions.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Push all crates onto the green targets before time runs out!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game. Race against the clock to push crates onto target locations in a minimalist, clean environment."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRID_SIZE = 40
        self.INTERP_SPEED = 0.25 # Speed for smooth visual movement

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (30, 45, 60)
        self.COLOR_WALL = (80, 90, 100)
        self.COLOR_WALL_TOP = (110, 120, 130)
        self.COLOR_TARGET = (40, 80, 60)
        self.COLOR_TARGET_LIT = (60, 180, 120)
        self.COLOR_CRATE = (139, 69, 19)
        self.COLOR_CRATE_BORDER = (90, 45, 10)
        self.COLOR_CRATE_LIT_BORDER = (120, 255, 180)
        self.COLOR_PLAYER = (255, 200, 0)
        self.COLOR_PLAYER_EYE = (0, 0, 0)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_TIMER_WARN = (255, 80, 80)

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
        self.font_ui = pygame.font.SysFont("Arial", 18)
        self.font_timer = pygame.font.SysFont("Arial", 26, bold=True)
        
        # --- Game State (initialized in reset) ---
        self.level_layout = []
        self.walls = set()
        self.targets = []
        self.start_player_pos = []
        self.start_crate_pos = []
        self.grid_offset = (0, 0)
        self.player_pos = []
        self.crates = []
        self.player_visual_pos = []
        self.crates_visual_pos = []
        self.player_orientation = (0, -1)
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = 0

        self._define_level()
        self.reset()

        self.validate_implementation()
    
    def _define_level(self):
        """Defines the static layout of the game level."""
        self.level_layout = [
            "WWWWWWWWWWWWWWWW",
            "W              W",
            "W P C        T W",
            "W   W WWWWWW   W",
            "W C W      W T W",
            "W   W      W   W",
            "W T WWWWWW W C W",
            "W              W",
            "WWWWWWWWWWWWWWWW",
        ]
        grid_h = len(self.level_layout)
        grid_w = len(self.level_layout[0])
        offset_x = (self.WIDTH - grid_w * self.GRID_SIZE) // 2
        offset_y = (self.HEIGHT - grid_h * self.GRID_SIZE) // 2
        self.grid_offset = (offset_x, offset_y)

        self.walls = set()
        self.targets = []
        self.start_crate_pos = []
        for r, row in enumerate(self.level_layout):
            for c, char in enumerate(row):
                pos = (c, r)
                if char == 'W':
                    self.walls.add(pos)
                elif char == 'P':
                    self.start_player_pos = list(pos)
                elif char == 'C':
                    self.start_crate_pos.append(list(pos))
                elif char == 'T':
                    self.targets.append(list(pos))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = 30 * self.FPS

        self.player_pos = list(self.start_player_pos)
        self.crates = [list(pos) for pos in self.start_crate_pos]

        def grid_to_pixel(pos):
            return [
                pos[0] * self.GRID_SIZE + self.grid_offset[0] + self.GRID_SIZE / 2,
                pos[1] * self.GRID_SIZE + self.grid_offset[1] + self.GRID_SIZE / 2,
            ]

        self.player_visual_pos = grid_to_pixel(self.player_pos)
        self.crates_visual_pos = [grid_to_pixel(c) for c in self.crates]
        self.player_orientation = (0, -1)
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_left -= 1
        
        movement = action[0]
        
        prev_on_target_indices = self._get_crates_on_targets()
        prev_dist = self._calculate_total_crate_dist()

        self._handle_movement(movement)
        
        terminated = self._check_termination()
        self.game_over = terminated
        
        reward = self._calculate_reward(prev_on_target_indices, prev_dist, terminated)
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 0:
            return

        move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # Up, Down, Left, Right
        dx, dy = move_map[movement]
        self.player_orientation = (dx, dy)

        next_player_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)

        if next_player_pos in self.walls:
            return # Bump into wall

        crate_indices = [i for i, c in enumerate(self.crates) if tuple(c) == next_player_pos]

        if crate_indices:
            crate_idx = crate_indices[0]
            next_crate_pos = (self.crates[crate_idx][0] + dx, self.crates[crate_idx][1] + dy)

            if next_crate_pos in self.walls or any(tuple(c) == next_crate_pos for c in self.crates):
                return # Crate is blocked

            # Move crate and player
            self.crates[crate_idx][0] += dx
            self.crates[crate_idx][1] += dy
            self.player_pos[0] += dx
            self.player_pos[1] += dy
        else:
            # Move player
            self.player_pos[0] += dx
            self.player_pos[1] += dy

    def _calculate_total_crate_dist(self):
        """Calculates sum of Manhattan distances from each crate to the nearest free target."""
        total_dist = 0
        unoccupied_targets = [tuple(t) for t in self.targets if tuple(t) not in [tuple(c) for c in self.crates]]
        
        for crate_pos in self.crates:
            if tuple(crate_pos) in {tuple(t) for t in self.targets}:
                continue # Crate is already on a target, its distance is 0.
            
            if not unoccupied_targets:
                # This case shouldn't happen if #crates <= #targets, but is a good safeguard.
                # Assign a large penalty distance.
                total_dist += 100 
                continue

            min_dist = float('inf')
            for target_pos in unoccupied_targets:
                dist = abs(crate_pos[0] - target_pos[0]) + abs(crate_pos[1] - target_pos[1])
                min_dist = min(min_dist, dist)
            total_dist += min_dist
        return total_dist

    def _get_crates_on_targets(self):
        """Returns a set of indices for crates that are on any target."""
        target_set = {tuple(t) for t in self.targets}
        on_target_indices = {i for i, c in enumerate(self.crates) if tuple(c) in target_set}
        return on_target_indices

    def _calculate_reward(self, prev_on_target_indices, prev_dist, terminated):
        reward = -0.1  # Time penalty

        current_on_target_indices = self._get_crates_on_targets()
        newly_placed_indices = current_on_target_indices - prev_on_target_indices

        if newly_placed_indices:
            reward += len(newly_placed_indices) * 10.0
            # Add particles for visual feedback
            for idx in newly_placed_indices:
                px, py = self.crates_visual_pos[idx]
                self._add_particles(px, py, 20, self.COLOR_TARGET_LIT)
                # SFX: Crate placed on target

        # Reward for moving closer/further
        current_dist = self._calculate_total_crate_dist()
        dist_change = prev_dist - current_dist
        reward += dist_change

        # Win bonus
        if terminated and self.time_left > 0:
            reward += 50.0
            # SFX: Level complete

        return reward

    def _check_termination(self):
        if self.time_left <= 0:
            return True
        if len(self._get_crates_on_targets()) == len(self.crates):
            return True
        return False

    def _update_visuals(self):
        # Interpolate player
        target_px, target_py = [p * self.GRID_SIZE + self.grid_offset[0] + self.GRID_SIZE / 2 for p in self.player_pos]
        self.player_visual_pos[0] += (target_px - self.player_visual_pos[0]) * self.INTERP_SPEED
        self.player_visual_pos[1] += (target_py - self.player_visual_pos[1]) * self.INTERP_SPEED

        # Interpolate crates
        for i, crate in enumerate(self.crates):
            target_cx, target_cy = [p * self.GRID_SIZE + self.grid_offset[0] + self.GRID_SIZE / 2 for p in crate]
            self.crates_visual_pos[i][0] += (target_cx - self.crates_visual_pos[i][0]) * self.INTERP_SPEED
            self.crates_visual_pos[i][1] += (target_cy - self.crates_visual_pos[i][1]) * self.INTERP_SPEED

        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] -= 0.1
            if p['life'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _add_particles(self, x, y, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': random.uniform(2, 5),
                'life': random.randint(10, 20),
                'color': color
            })

    def _get_observation(self):
        self._update_visuals()
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid lines
        for r in range(len(self.level_layout) + 1):
            y = self.grid_offset[1] + r * self.GRID_SIZE
            start_pos = (self.grid_offset[0], y)
            end_pos = (self.WIDTH - self.grid_offset[0], y)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
        for c in range(len(self.level_layout[0]) + 1):
            x = self.grid_offset[0] + c * self.GRID_SIZE
            start_pos = (x, self.grid_offset[1])
            end_pos = (x, self.HEIGHT - self.grid_offset[1])
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)

        # Draw targets
        crate_positions_set = {tuple(c) for c in self.crates}
        for t_pos in self.targets:
            is_occupied = tuple(t_pos) in crate_positions_set
            color = self.COLOR_TARGET_LIT if is_occupied else self.COLOR_TARGET
            px, py = [p * self.GRID_SIZE + self.grid_offset[0] for p in t_pos]
            pygame.draw.rect(self.screen, color, (px, py, self.GRID_SIZE, self.GRID_SIZE))

        # Draw walls
        for w_pos in self.walls:
            px, py = [p * self.GRID_SIZE + self.grid_offset[0] for p in w_pos]
            pygame.draw.rect(self.screen, self.COLOR_WALL, (px, py, self.GRID_SIZE, self.GRID_SIZE))
            pygame.draw.rect(self.screen, self.COLOR_WALL_TOP, (px, py, self.GRID_SIZE, 4))

        # Draw crates
        crates_on_targets_indices = self._get_crates_on_targets()
        for i, c_vis_pos in enumerate(self.crates_visual_pos):
            on_target = i in crates_on_targets_indices
            border_color = self.COLOR_CRATE_LIT_BORDER if on_target else self.COLOR_CRATE_BORDER
            size = self.GRID_SIZE * 0.8
            half_size = size / 2
            rect = pygame.Rect(c_vis_pos[0] - half_size, c_vis_pos[1] - half_size, size, size)
            pygame.draw.rect(self.screen, self.COLOR_CRATE, rect, border_radius=4)
            pygame.draw.rect(self.screen, border_color, rect, width=3, border_radius=4)

        # Draw player
        p_vis_x, p_vis_y = self.player_visual_pos
        radius = int(self.GRID_SIZE * 0.4)
        pygame.gfxdraw.filled_circle(self.screen, int(p_vis_x), int(p_vis_y), radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, int(p_vis_x), int(p_vis_y), radius, self.COLOR_PLAYER)
        
        eye_x = p_vis_x + self.player_orientation[0] * radius * 0.5
        eye_y = p_vis_y + self.player_orientation[1] * radius * 0.5
        pygame.gfxdraw.filled_circle(self.screen, int(eye_x), int(eye_y), int(radius * 0.25), self.COLOR_PLAYER_EYE)

        # Draw particles
        for p in self.particles:
            radius = int(p['radius'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, p['color'])

    def _render_ui(self):
        score_surf = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (15, 10))

        completed_count = len(self._get_crates_on_targets())
        total_crates = len(self.crates)
        completed_surf = self.font_ui.render(f"COMPLETED: {completed_count} / {total_crates}", True, self.COLOR_TEXT)
        self.screen.blit(completed_surf, (15, 35))

        time_sec = self.time_left / self.FPS
        timer_color = self.COLOR_TIMER_WARN if time_sec < 5 and self.time_left > 0 else self.COLOR_TEXT
        timer_text = f"{max(0, time_sec):.1f}"
        timer_surf = self.font_timer.render(timer_text, True, timer_color)
        timer_rect = timer_surf.get_rect(topright=(self.WIDTH - 15, 10))
        self.screen.blit(timer_surf, timer_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left_seconds": self.time_left / self.FPS,
            "crates_on_target": len(self._get_crates_on_targets()),
        }

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to test the environment with human controls.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Sokoban Racer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    total_reward = 0
    
    running = True
    while running:
        action = env.action_space.no_op()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        if keys[pygame.K_r]:
            obs, info = env.reset()
            total_reward = 0
            terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
    env.close()