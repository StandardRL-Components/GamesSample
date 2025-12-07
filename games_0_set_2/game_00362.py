
# Generated: 2025-08-27T13:25:26.149687
# Source Brief: brief_00362.md
# Brief Index: 362

        
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

    # Must be a short, user-facing control string:
    user_guide = "Controls: Use arrow keys to move. Push all brown blocks onto the gold goals before time runs out."

    # Must be a short, user-facing description of the game:
    game_description = "Solve isometric puzzles by pushing all blocks onto the goals. Each level is more complex and has less time."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1500 # 50 seconds at 30 FPS

        # World constants
        self.GRID_W, self.GRID_H = 12, 10
        self.TILE_W, self.TILE_H = 48, 24
        self.ISO_OFFSET_X = self.WIDTH // 2
        self.ISO_OFFSET_Y = 80
        self.ANIMATION_SPEED = 0.2 # Higher is faster

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_WALL = (100, 110, 130)
        self.COLOR_WALL_TOP = (120, 130, 150)
        self.COLOR_GOAL = (255, 215, 0)
        self.COLOR_GOAL_INACTIVE = (80, 70, 20)
        self.COLOR_BLOCK = (139, 69, 19)
        self.COLOR_BLOCK_TOP = (160, 82, 45)
        self.COLOR_PLAYER = (0, 120, 255)
        self.COLOR_PLAYER_TOP = (50, 150, 255)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_TIMER_WARN = (255, 100, 100)

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # Initialize state variables
        self.player_pos = None
        self.player_visual_pos = None
        self.blocks = None
        self.blocks_visual_pos = None
        self.goals = None
        self.walls = None
        self.level = None
        self.initial_time = None
        self.time_remaining = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.action_cooldown = 0
        self.particles = []

        self.reset()
        # self.validate_implementation() # Optional: uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.level = 1
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles.clear()
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_remaining -= 1 / self.FPS
        self.action_cooldown = max(0, self.action_cooldown - 1)
        
        reward = 0
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Process movement if not on cooldown
        if movement != 0 and self.action_cooldown == 0:
            reward += self._handle_movement(movement)
            # Start cooldown after a successful action to prevent spamming
            if reward != 0:
                self.action_cooldown = int(self.FPS * 0.2) # 0.2 second cooldown

        self.score += reward
        
        terminated, term_reward = self._check_termination()
        self.game_over = terminated
        self.score += term_reward
        reward += term_reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self):
        self.walls = set()
        self.goals = set()
        self.blocks = []
        
        # Border walls
        for i in range(self.GRID_W):
            self.walls.add((i, -1))
            self.walls.add((i, self.GRID_H))
        for i in range(self.GRID_H):
            self.walls.add((-1, i))
            self.walls.add((self.GRID_W, i))

        # Level 1: Hardcoded simple layout
        if self.level == 1:
            self.initial_time = 90.0
            self.goals = {(3, 4), (4, 4)}
            self.blocks = [(3, 2), (4, 2)]
            self.player_pos = (3, 6)
            # Add a few internal walls
            self.walls.update([(5,2), (5,3), (5,4), (5,5), (5,6)])
        else: # Procedural generation for subsequent levels
            self.initial_time = max(30.0, 90.0 - (self.level - 1) * 5)
            num_blocks = min(self.level + 1, (self.GRID_W - 4) * (self.GRID_H - 4) // 4)

            # Generate valid empty spaces
            valid_spaces = []
            for r in range(1, self.GRID_H - 1):
                for c in range(1, self.GRID_W - 1):
                    valid_spaces.append((c, r))
            
            self.np_random.shuffle(valid_spaces)
            
            # Place goals
            for _ in range(num_blocks):
                self.goals.add(valid_spaces.pop())
            
            # Place blocks
            for _ in range(num_blocks):
                self.blocks.append(valid_spaces.pop())
            
            # Place player
            self.player_pos = valid_spaces.pop()

        self.time_remaining = self.initial_time
        
        # Initialize visual positions
        self.player_visual_pos = self._cart_to_iso(*self.player_pos)
        self.blocks_visual_pos = [self._cart_to_iso(*pos) for pos in self.blocks]

    def _handle_movement(self, movement):
        moves = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # Up, Down, Left, Right
        dx, dy = moves[movement]
        
        px, py = self.player_pos
        next_pos = (px + dx, py + dy)

        # Case 1: Move into empty space
        if next_pos not in self.walls and next_pos not in [tuple(b) for b in self.blocks]:
            self.player_pos = next_pos
            return 0 # No immediate reward for just moving

        # Case 2: Attempt to push a block
        if next_pos in [tuple(b) for b in self.blocks]:
            block_idx = [tuple(b) for b in self.blocks].index(next_pos)
            block_pos = self.blocks[block_idx]
            next_block_pos = (block_pos[0] + dx, block_pos[1] + dy)

            if next_block_pos not in self.walls and next_block_pos not in [tuple(b) for b in self.blocks]:
                # Successful push
                old_block_pos = self.blocks[block_idx]
                self.blocks[block_idx] = list(next_block_pos)
                self.player_pos = next_pos
                return self._calculate_push_reward(old_block_pos, next_block_pos)
        
        return 0 # Invalid move

    def _calculate_push_reward(self, old_pos, new_pos):
        reward = 0
        was_on_goal = old_pos in self.goals
        is_on_goal = new_pos in self.goals

        if not was_on_goal and is_on_goal:
            reward += 1.0
            # Sound: sfx_block_on_goal.wav
            self._create_particles(self._cart_to_iso(*new_pos), self.COLOR_GOAL)
        elif was_on_goal and not is_on_goal:
            reward -= 1.0 # Penalize moving a block off a goal

        # Proximity reward
        try:
            # Find closest goal to this block
            dist_to_goals = [math.hypot(g[0]-new_pos[0], g[1]-new_pos[1]) for g in self.goals]
            closest_goal_dist_after = min(dist_to_goals)
            
            dist_to_goals_old = [math.hypot(g[0]-old_pos[0], g[1]-old_pos[1]) for g in self.goals]
            closest_goal_dist_before = min(dist_to_goals_old)

            if closest_goal_dist_after < closest_goal_dist_before:
                reward += 0.1
            else:
                reward -= 0.01
        except ValueError: # No goals
            pass
        
        return reward

    def _check_termination(self):
        # Win condition: all blocks on goals
        blocks_on_goals = sum(1 for b in self.blocks if tuple(b) in self.goals)
        if blocks_on_goals == len(self.goals) and len(self.goals) > 0:
            win_reward = 50 * (self.time_remaining / self.initial_time)
            # Sound: sfx_level_complete.wav
            self.level += 1
            self._generate_level() # Transition to next level
            return False, win_reward # Don't terminate, just give reward and go to next level

        # Lose condition: time out
        if self.time_remaining <= 0:
            return True, -100.0
        
        # Lose condition: max steps
        if self.steps >= self.MAX_STEPS:
            return True, -50.0

        return False, 0

    def _get_observation(self):
        self._update_animations()
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
            "time_remaining": self.time_remaining,
            "blocks_on_goals": sum(1 for b in self.blocks if tuple(b) in self.goals),
            "total_goals": len(self.goals)
        }

    def _cart_to_iso(self, x, y):
        iso_x = (x - y) * (self.TILE_W / 2) + self.ISO_OFFSET_X
        iso_y = (x + y) * (self.TILE_H / 2) + self.ISO_OFFSET_Y
        return iso_x, iso_y

    def _update_animations(self):
        # Player visual position
        target_px, target_py = self._cart_to_iso(*self.player_pos)
        self.player_visual_pos = (
            self.player_visual_pos[0] + (target_px - self.player_visual_pos[0]) * self.ANIMATION_SPEED,
            self.player_visual_pos[1] + (target_py - self.player_visual_pos[1]) * self.ANIMATION_SPEED
        )

        # Blocks visual positions
        for i, block_pos in enumerate(self.blocks):
            target_bx, target_by = self._cart_to_iso(*block_pos)
            self.blocks_visual_pos[i] = (
                self.blocks_visual_pos[i][0] + (target_bx - self.blocks_visual_pos[i][0]) * self.ANIMATION_SPEED,
                self.blocks_visual_pos[i][1] + (target_by - self.blocks_visual_pos[i][1]) * self.ANIMATION_SPEED
            )
            
        # Update particles
        for p in self.particles[:]:
            p[0] += p[2] # pos_x += vel_x
            p[1] += p[3] # pos_y += vel_y
            p[3] += 0.1 # gravity
            p[4] -= 1 # lifetime
            if p[4] <= 0:
                self.particles.remove(p)

    def _draw_iso_rect(self, surface, color, x, y, w, h):
        points = [
            (x, y - h),
            (x + w, y),
            (x, y + h),
            (x - w, y)
        ]
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _draw_iso_cube(self, surface, pos, color_top, color_side, height=16):
        x, y = pos
        w, h = self.TILE_W / 2, self.TILE_H / 2
        
        top_points = [(x, y - h), (x + w, y), (x, y + h), (x - w, y)]
        side1_points = [(x - w, y), (x, y + h), (x, y + h + height), (x - w, y + height)]
        side2_points = [(x + w, y), (x, y + h), (x, y + h + height), (x + w, y + height)]

        pygame.gfxdraw.aapolygon(surface, side1_points, color_side)
        pygame.gfxdraw.filled_polygon(surface, side1_points, color_side)
        pygame.gfxdraw.aapolygon(surface, side2_points, color_side)
        pygame.gfxdraw.filled_polygon(surface, side2_points, color_side)
        pygame.gfxdraw.aapolygon(surface, top_points, color_top)
        pygame.gfxdraw.filled_polygon(surface, top_points, color_top)
        
    def _create_particles(self, pos, color):
        for _ in range(20):
            self.particles.append([
                pos[0], pos[1], # x, y
                self.np_random.uniform(-2, 2), self.np_random.uniform(-3, -1), # vx, vy
                self.np_random.integers(20, 40), # lifetime
                color
            ])

    def _render_game(self):
        # Prepare renderables
        render_list = []

        # Floor and Goals
        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                iso_x, iso_y = self._cart_to_iso(c, r)
                color = self.COLOR_GOAL if (c, r) in self.goals else self.COLOR_GRID
                if (c, r) in self.goals and (c,r) in [tuple(b) for b in self.blocks]:
                    color = self.COLOR_GOAL # Keep goal color if block is on it
                
                self._draw_iso_rect(self.screen, color, iso_x, iso_y, self.TILE_W / 2, self.TILE_H / 2)

        # Walls, Blocks, Player
        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                if (c, r) in self.walls:
                    render_list.append(('wall', (c, r), r))
        
        for i, pos in enumerate(self.blocks):
            render_list.append(('block', i, pos[1]))
            
        render_list.append(('player', None, self.player_pos[1]))
        
        # Sort by y-grid-coordinate for correct occlusion
        render_list.sort(key=lambda item: item[2] + (0.5 if item[0] == 'player' else 0))

        for item_type, data, _ in render_list:
            if item_type == 'wall':
                iso_pos = self._cart_to_iso(*data)
                self._draw_iso_cube(self.screen, iso_pos, self.COLOR_WALL_TOP, self.COLOR_WALL)
            elif item_type == 'block':
                visual_pos = self.blocks_visual_pos[data]
                self._draw_iso_cube(self.screen, visual_pos, self.COLOR_BLOCK_TOP, self.COLOR_BLOCK)
            elif item_type == 'player':
                self._draw_iso_cube(self.screen, self.player_visual_pos, self.COLOR_PLAYER_TOP, self.COLOR_PLAYER, height=20)
                
        # Render particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p[5], (int(p[0]), int(p[1])), int(p[4] / 10))

    def _render_ui(self):
        # Level display
        level_text = self.font_main.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 10))
        
        # Timer display
        time_str = f"{max(0, self.time_remaining):.1f}"
        time_color = self.COLOR_TIMER_WARN if self.time_remaining < 10 else self.COLOR_TEXT
        time_text = self.font_main.render(time_str, True, time_color)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)
        
        # Goals display
        blocks_on_goals = sum(1 for b in self.blocks if tuple(b) in self.goals)
        total_goals = len(self.goals)
        goals_text = self.font_main.render(f"GOALS: {blocks_on_goals}/{total_goals}", True, self.COLOR_TEXT)
        self.screen.blit(goals_text, (10, self.HEIGHT - 34))

        # Score display
        score_text = self.font_main.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(bottomright=(self.WIDTH - 10, self.HEIGHT - 10))
        self.screen.blit(score_text, score_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to run and visualize the game
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- For human play ---
    # env.reset()
    # screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    # pygame.display.set_caption("Sokoban ISO")
    # clock = pygame.time.Clock()
    # running = True
    # while running:
    #     action = np.array([0, 0, 0]) # Default no-op
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_UP]:
    #         action[0] = 1
    #     elif keys[pygame.K_DOWN]:
    #         action[0] = 2
    #     elif keys[pygame.K_LEFT]:
    #         action[0] = 3
    #     elif keys[pygame.K_RIGHT]:
    #         action[0] = 4
        
    #     obs, reward, terminated, truncated, info = env.step(action)
        
    #     # Draw the observation to the display
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     if terminated:
    #         print(f"Game Over! Final Score: {info['score']:.2f}, Level: {info['level']}")
    #         env.reset()
    #         pygame.time.wait(2000)

    #     clock.tick(env.FPS)
    # env.close()

    # --- For validation and testing ---
    env.validate_implementation()
    
    obs, info = env.reset()
    print("Initial Info:", info)
    
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if (i+1) % 20 == 0:
            print(f"Step {i+1}: Reward={reward:.2f}, Info={info}")
        if terminated:
            print("Episode finished after {} steps.".format(i+1))
            obs, info = env.reset()
    
    env.close()