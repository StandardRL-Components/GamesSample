
# Generated: 2025-08-28T07:01:58.213008
# Source Brief: brief_03115.md
# Brief Index: 3115

        
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
        "Controls: ↑↓←→ to move your character and push blocks."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Push pixel blocks to their goal locations within a time limit across multiple stages."
    )

    # Frames auto-advance at 30fps for smooth, real-time gameplay.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.TILE_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.TILE_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.TILE_SIZE
        self.NUM_BLOCKS = 10
        self.FPS = 30
        self.STAGE_TIME = 60.0
        self.MAX_STAGES = 3
        
        # --- Colors ---
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (40, 40, 55)
        self.COLOR_PLAYER = (255, 70, 70)
        self.COLOR_PLAYER_INNER = (255, 150, 150)
        self.COLOR_BLOCK = (60, 120, 255)
        self.COLOR_BLOCK_INNER = (130, 180, 255)
        self.COLOR_GOAL = (255, 220, 50)
        self.COLOR_OBSTACLE = (100, 100, 110)
        self.COLOR_OBSTACLE_INNER = (140, 140, 150)
        self.COLOR_BLOCK_IN_GOAL = (60, 255, 120)
        self.COLOR_BLOCK_IN_GOAL_INNER = (130, 255, 180)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TIMER_WARN = (255, 100, 100)

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        
        # --- Game State (initialized in reset) ---
        self.stage = 1
        self.score = 0
        self.timer = 0.0
        self.game_over = False
        self.win = False
        self.total_steps = 0
        
        self.player_pos = [0, 0]
        self.player_render_pos = [0.0, 0.0]
        self.blocks = []
        self.goals = []
        self.obstacles = []
        self.particles = []

        # Initialize state variables
        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def _setup_stage(self):
        """Initializes the layout for the current stage."""
        self.timer = self.STAGE_TIME
        self.player_pos = [0, 0]
        self.blocks.clear()
        self.goals.clear()
        self.obstacles.clear()
        self.particles.clear()

        num_obstacles = {1: 0, 2: 2, 3: 4}.get(self.stage, 4)

        # Generate all possible spawn locations with a 1-tile margin
        possible_coords = []
        for x in range(1, self.GRID_WIDTH - 1):
            for y in range(1, self.GRID_HEIGHT - 1):
                possible_coords.append([x, y])
        
        self.np_random.shuffle(possible_coords)

        # Place entities, ensuring no overlap
        entity_count = 1 + self.NUM_BLOCKS + self.NUM_BLOCKS + num_obstacles
        if entity_count > len(possible_coords):
            raise ValueError("Not enough space on the grid to place all entities.")
        
        spawn_points = possible_coords[:entity_count]
        
        self.player_pos = spawn_points.pop()
        self.player_render_pos = [float(self.player_pos[0] * self.TILE_SIZE), float(self.player_pos[1] * self.TILE_SIZE)]

        for _ in range(self.NUM_BLOCKS):
            self.goals.append(spawn_points.pop())

        for _ in range(self.NUM_BLOCKS):
            pos = spawn_points.pop()
            self.blocks.append({
                'pos': pos,
                'render_pos': [float(pos[0] * self.TILE_SIZE), float(pos[1] * self.TILE_SIZE)],
                'in_goal': tuple(pos) in self.goals,
            })

        for _ in range(num_obstacles):
            self.obstacles.append(spawn_points.pop())

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.stage = 1
        self.score = 0
        self.total_steps = 0
        self.game_over = False
        self.win = False
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0.0
        self.total_steps += 1
        self.timer = max(0, self.timer - 1.0 / self.FPS)

        movement = action[0]
        
        if not self.game_over:
            # --- Player Movement and Pushing Logic ---
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right

            if dx != 0 or dy != 0:
                next_player_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]
                
                # Check boundaries
                if not (0 <= next_player_pos[0] < self.GRID_WIDTH and 0 <= next_player_pos[1] < self.GRID_HEIGHT):
                    pass # Wall collision
                # Check obstacles
                elif next_player_pos in self.obstacles:
                    pass # Obstacle collision
                else:
                    # Check for block collision
                    block_to_push = None
                    for block in self.blocks:
                        if block['pos'] == next_player_pos:
                            block_to_push = block
                            break
                    
                    if block_to_push:
                        next_block_pos = [block_to_push['pos'][0] + dx, block_to_push['pos'][1] + dy]
                        
                        # Check if block's next position is valid
                        is_next_pos_free = (0 <= next_block_pos[0] < self.GRID_WIDTH and
                                            0 <= next_block_pos[1] < self.GRID_HEIGHT and
                                            next_block_pos not in self.obstacles and
                                            next_block_pos not in [b['pos'] for b in self.blocks])
                        
                        if is_next_pos_free:
                            # --- Distance-based Reward ---
                            old_dist = min(self._manhattan_distance(block_to_push['pos'], g) for g in self.goals)
                            new_dist = min(self._manhattan_distance(next_block_pos, g) for g in self.goals)
                            if new_dist < old_dist:
                                reward += 0.1
                            else:
                                reward -= 0.1

                            block_to_push['pos'] = next_block_pos
                            self.player_pos = next_player_pos
                            # # Sound: block_push.wav
                            self._create_particles(next_player_pos[0] * self.TILE_SIZE, next_player_pos[1] * self.TILE_SIZE)
                        else:
                            pass # Block push failed
                    else:
                        # No block, just move player
                        self.player_pos = next_player_pos

            # --- Update Block 'in_goal' Status and Rewards ---
            stage_complete = True
            for block in self.blocks:
                is_in_goal = block['pos'] in self.goals
                if is_in_goal and not block['in_goal']:
                    reward += 1.0
                    self.score += 10
                    # # Sound: goal_achieved.wav
                block['in_goal'] = is_in_goal
                if not is_in_goal:
                    stage_complete = False

            # --- Stage Progression ---
            if stage_complete:
                reward += 10.0
                self.score += 100
                self.stage += 1
                if self.stage > self.MAX_STAGES:
                    self.win = True
                    self.game_over = True
                    reward += 100.0
                    self.score += 1000
                    # # Sound: game_win.wav
                else:
                    self._setup_stage()
                    # # Sound: stage_clear.wav

        # --- Update Animations and Particles ---
        self._update_render_positions()
        self._update_particles()

        # --- Termination Conditions ---
        if self.timer <= 0 and not self.game_over:
            self.game_over = True
            # # Sound: game_over.wav
        
        terminated = self.game_over or self.win

        # Update final score based on remaining time if won
        if terminated and self.win:
            self.score += int(self.timer * 10)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _lerp(self, start, end, t):
        return start + t * (end - start)

    def _update_render_positions(self):
        """Smoothly interpolates render positions towards grid positions."""
        lerp_factor = 0.5
        
        # Player
        target_x = self.player_pos[0] * self.TILE_SIZE
        target_y = self.player_pos[1] * self.TILE_SIZE
        self.player_render_pos[0] = self._lerp(self.player_render_pos[0], target_x, lerp_factor)
        self.player_render_pos[1] = self._lerp(self.player_render_pos[1], target_y, lerp_factor)

        # Blocks
        for block in self.blocks:
            target_x = block['pos'][0] * self.TILE_SIZE
            target_y = block['pos'][1] * self.TILE_SIZE
            block['render_pos'][0] = self._lerp(block['render_pos'][0], target_x, lerp_factor)
            block['render_pos'][1] = self._lerp(block['render_pos'][1], target_y, lerp_factor)

    def _create_particles(self, x, y):
        for _ in range(10):
            self.particles.append({
                'pos': [x + self.TILE_SIZE / 2, y + self.TILE_SIZE / 2],
                'vel': [self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)],
                'life': self.np_random.uniform(10, 20),
                'color': self.COLOR_BLOCK_INNER,
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Draw goals
        for gx, gy in self.goals:
            rect = pygame.Rect(gx * self.TILE_SIZE, gy * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.gfxdraw.box(self.screen, rect, (*self.COLOR_GOAL, 50))
            pygame.gfxdraw.rectangle(self.screen, rect, self.COLOR_GOAL)

        # Draw obstacles
        for ox, oy in self.obstacles:
            self._draw_beveled_rect(
                int(ox * self.TILE_SIZE), int(oy * self.TILE_SIZE),
                self.TILE_SIZE, self.TILE_SIZE,
                self.COLOR_OBSTACLE, self.COLOR_OBSTACLE_INNER
            )

        # Draw blocks
        for block in self.blocks:
            color, inner_color = (self.COLOR_BLOCK_IN_GOAL, self.COLOR_BLOCK_IN_GOAL_INNER) if block['in_goal'] else (self.COLOR_BLOCK, self.COLOR_BLOCK_INNER)
            self._draw_beveled_rect(
                int(block['render_pos'][0]), int(block['render_pos'][1]),
                self.TILE_SIZE, self.TILE_SIZE,
                color, inner_color
            )
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20))))
            radius = int(p['life'] / 4)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, (*p['color'], alpha))

        # Draw player
        self._draw_beveled_rect(
            int(self.player_render_pos[0]), int(self.player_render_pos[1]),
            self.TILE_SIZE, self.TILE_SIZE,
            self.COLOR_PLAYER, self.COLOR_PLAYER_INNER
        )
    
    def _draw_beveled_rect(self, x, y, w, h, color_outer, color_inner):
        """Draws a rect with an inner bevel for a 3D-ish look."""
        margin = max(1, int(w * 0.15))
        pygame.draw.rect(self.screen, color_outer, (x, y, w, h))
        pygame.draw.rect(self.screen, color_inner, (x + margin, y + margin, w - 2 * margin, h - 2 * margin))

    def _render_ui(self):
        # Draw UI Panel
        panel_rect = pygame.Rect(0, 0, self.WIDTH, 40)
        pygame.draw.rect(self.screen, (0,0,0,150), panel_rect)

        # Stage Text
        stage_text = self.font_large.render(f"Stage: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, 8))

        # Score Text
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(centerx=self.WIDTH / 2, y=8)
        self.screen.blit(score_text, score_rect)

        # Timer Text
        timer_color = self.COLOR_TIMER_WARN if self.timer < 10 and int(self.timer*2) % 2 == 0 else self.COLOR_TEXT
        timer_text = self.font_large.render(f"Time: {self.timer:.1f}", True, timer_color)
        timer_rect = timer_text.get_rect(right=self.WIDTH - 10, y=8)
        self.screen.blit(timer_text, timer_rect)

        # Game Over / Win Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "TIME'S UP!"
            msg_surf = self.font_large.render(message, True, self.COLOR_GOAL if self.win else self.COLOR_TIMER_WARN)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
            self.screen.blit(msg_surf, msg_rect)

            final_score_surf = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 20))
            self.screen.blit(final_score_surf, final_score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.total_steps,
            "stage": self.stage,
            "timer": self.timer
        }

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Pusher")
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

    env.close()