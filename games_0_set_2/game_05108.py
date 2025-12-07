
# Generated: 2025-08-28T03:59:19.432696
# Source Brief: brief_05108.md
# Brief Index: 5108

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to hop between adjacent tiles. "
        "Reach the blue tile before time runs out!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric skill-based platformer. Hop across procedurally generated tile paths to reach the "
        "goal. Green tiles are safe, yellow are neutral, and red are risky. Be quick, the clock is ticking!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 20)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- Visuals ---
        self.COLOR_BG_TOP = (4, 10, 18)
        self.COLOR_BG_BOTTOM = (12, 24, 48)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_SHADOW = (0, 0, 0, 100)
        self.TILE_COLORS = {
            "safe": (50, 205, 50),    # LimeGreen
            "medium": (255, 215, 0),   # Gold
            "high": (220, 20, 60),     # Crimson
            "end": (30, 144, 255),     # DodgerBlue
            "start": (150, 150, 150)   # Grey for start
        }
        self.TILE_W = 60
        self.TILE_H = 30
        self.TILE_DEPTH = 20
        self.ORIGIN_X = 640 // 2
        self.ORIGIN_Y = 100

        # --- Game State ---
        # Persistent across resets
        self.current_stage = 1
        self.max_stages = 3

        # Reset per episode
        self.player_grid_pos = None
        self.player_hop_info = None # {start_screen, end_screen, is_fall}
        self.tiles = None
        self.end_pos = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win_message = ""
        self.start_time = None
        self.time_remaining = None
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.game_over and "STAGE CLEAR" in self.win_message:
            # Progress to the next stage naturally
            pass
        elif self.game_over and "YOU WIN" in self.win_message:
            self.current_stage = 1 # Loop back to start after winning
        elif options and 'stage' in options:
            self.current_stage = options['stage']
        
        # Reset episode-specific state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.start_time = pygame.time.get_ticks()
        self.time_remaining = 60.0

        self.player_grid_pos = [0, 0]
        self.player_hop_info = None

        self._generate_level()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean (unused)
        shift_held = action[2] == 1  # Boolean (unused)
        
        reward = 0
        terminated = False
        
        # Update time; it ticks down regardless of action
        elapsed_seconds = (pygame.time.get_ticks() - self.start_time) / 1000.0
        self.time_remaining = max(0, 60.0 - elapsed_seconds)
        
        if self.time_remaining <= 0:
            reward = -10 # Timeout penalty
            terminated = True
            self.game_over = True
            self.win_message = "TIME UP"
        
        if movement != 0 and not terminated:
            start_pos = self.player_grid_pos.copy()
            target_pos = self.player_grid_pos.copy()
            
            # 1=up(iso), 2=down(iso), 3=left(iso), 4=right(iso)
            if movement == 1: target_pos[1] -= 1 # Up-Right on screen
            elif movement == 2: target_pos[1] += 1 # Down-Left on screen
            elif movement == 3: target_pos[0] -= 1 # Up-Left on screen
            elif movement == 4: target_pos[0] += 1 # Down-Right on screen

            target_tuple = tuple(target_pos)
            
            # Set up info for single-frame hop animation in _get_observation
            self.player_hop_info = {
                "start_screen": self._iso_to_screen(*start_pos),
                "end_screen": self._iso_to_screen(*target_pos),
                "is_fall": target_tuple not in self.tiles
            }

            if target_tuple in self.tiles:
                # Successful hop
                self.player_grid_pos = target_pos
                tile_type = self.tiles[target_tuple]['type']
                
                # Continuous feedback rewards
                if tile_type == 'safe': reward += 1
                elif tile_type == 'medium': reward += 0
                elif tile_type == 'high': reward += -1
                elif tile_type == 'end':
                    # Event-based rewards
                    reward += 5 # Reached end tile
                    reward += 50 # Stage completion bonus
                    terminated = True
                    self.game_over = True
                    if self.current_stage == self.max_stages:
                        self.win_message = "YOU WIN!"
                        # On next reset, it will loop to stage 1
                    else:
                        self.win_message = "STAGE CLEAR!"
                    # Stage progression
                    self.current_stage = min(self.max_stages + 1, self.current_stage + 1)

            else:
                # Fell off
                reward = -10
                terminated = True
                self.game_over = True
                self.win_message = "GAME OVER"
        
        self.steps += 1
        if self.steps >= 1000 and not terminated:
            terminated = True
            self.game_over = True
            self.win_message = "STEP LIMIT"

        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self._draw_gradient_background()
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage if self.current_stage <= self.max_stages else "Finished",
            "time_remaining": self.time_remaining
        }

    def _generate_level(self):
        self.tiles = {}
        path_len_base = 20
        high_risk_base = 0.1
        
        stage = min(self.current_stage, self.max_stages)

        # Difficulty scaling
        path_len = int(path_len_base * (1 + (stage - 1) * 0.25))
        high_risk_chance = high_risk_base + (stage - 1) * 0.1

        # Generate a guaranteed path using a biased random walk
        path = [(0, 0)]
        current_pos = [0, 0]
        self.tiles[(0, 0)] = {'type': 'start'}
        
        for _ in range(path_len):
            # Bias towards positive x and y to move away from origin
            direction = self.np_random.choice(np.array([[1,0], [-1,0], [0,1], [0,-1]]), p=[0.4, 0.1, 0.4, 0.1])
            next_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
            
            # Avoid immediate backtracking
            if len(path) > 1 and next_pos == path[-2]:
                 direction = self.np_random.choice(np.array([[1,0], [-1,0], [0,1], [0,-1]]), p=[0.4, 0.1, 0.4, 0.1])
                 next_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
            
            current_pos = list(next_pos)
            if tuple(current_pos) not in path:
                path.append(tuple(current_pos))

        # Populate tiles from path
        for i, pos in enumerate(path):
            if i == 0: continue
            if i == len(path) - 1:
                self.tiles[pos] = {'type': 'end'}
                self.end_pos = pos
            else:
                rand_val = self.np_random.random()
                if rand_val < high_risk_chance: tile_type = 'high'
                elif rand_val < high_risk_chance + 0.3: tile_type = 'medium'
                else: tile_type = 'safe'
                self.tiles[pos] = {'type': tile_type}

        # Add some branching dead-ends for complexity
        num_branches = self.np_random.integers(3, 6)
        for _ in range(num_branches):
            start_node_idx = self.np_random.integers(0, len(path))
            start_node = path[start_node_idx]
            branch_len = self.np_random.integers(2, 5)
            current_pos_branch = list(start_node)
            for _ in range(branch_len):
                direction = self.np_random.choice(np.array([[1,0], [-1,0], [0,1], [0,-1]]))
                current_pos_branch[0] += direction[0]
                current_pos_branch[1] += direction[1]
                pos_tuple = tuple(current_pos_branch)
                if pos_tuple not in self.tiles:
                    rand_val = self.np_random.random()
                    if rand_val < 0.4: tile_type = 'high'
                    elif rand_val < 0.8: tile_type = 'medium'
                    else: tile_type = 'safe'
                    self.tiles[pos_tuple] = {'type': tile_type}

    def _render_game(self):
        # Sort tiles by grid y, then x for correct painter's algorithm overlap
        sorted_tiles = sorted(self.tiles.items(), key=lambda item: (item[0][0] + item[0][1], item[0][1] - item[0][0]))

        for pos, tile_data in sorted_tiles:
            screen_pos = self._iso_to_screen(pos[0], pos[1])
            color = self.TILE_COLORS[tile_data['type']]
            self._draw_iso_cube(screen_pos, color)
        
        # --- Player Rendering ---
        player_z_offset = 0
        shadow_pos = self._iso_to_screen(*self.player_grid_pos)
        player_draw_pos = shadow_pos

        if self.player_hop_info:
            # Animate the hop in a single frame by showing it mid-air
            progress = 0.5 # Represents the peak of the jump
            
            # Interpolate screen position for the visual
            px = self.player_hop_info["start_screen"][0] + (self.player_hop_info["end_screen"][0] - self.player_hop_info["start_screen"][0]) * progress
            py = self.player_hop_info["start_screen"][1] + (self.player_hop_info["end_screen"][1] - self.player_hop_info["start_screen"][1]) * progress
            player_draw_pos = (px, py)
            shadow_pos = player_draw_pos
            
            # Calculate vertical hop arc
            hop_height = self.TILE_H * 1.5
            if self.player_hop_info["is_fall"]:
                player_z_offset = hop_height * progress * 2 # Falling down
            else:
                player_z_offset = -math.sin(math.pi * progress) * hop_height # Hopping up

            self.player_hop_info = None # Consume the animation info for next frame

        # Draw shadow
        shadow_size = self.TILE_W / 3
        shadow_surface = pygame.Surface((shadow_size, shadow_size / 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, self.COLOR_SHADOW, (0, 0, shadow_size, shadow_size/2))
        self.screen.blit(shadow_surface, (int(shadow_pos[0] - shadow_size / 2), int(shadow_pos[1] - shadow_size / 4)))
        
        # Draw player
        self._draw_iso_cube((player_draw_pos[0], player_draw_pos[1] + player_z_offset), self.COLOR_PLAYER, size_mod=0.5)

    def _render_ui(self):
        stage_display = self.current_stage if self.current_stage <= self.max_stages else "WIN"
        stage_text = self.font_small.render(f"Stage: {stage_display}", True, (255, 255, 255))
        self.screen.blit(stage_text, (10, 10))
        
        score_text = self.font_small.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 35))

        time_color = (255, 255, 255) if self.time_remaining > 10 else (255, 100, 100)
        time_str = f"Time: {self.time_remaining:.1f}"
        time_text = self.font_small.render(time_str, True, time_color)
        self.screen.blit(time_text, (640 - time_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_message, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(320, 200))
            self.screen.blit(end_text, text_rect)

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.ORIGIN_X + (grid_x - grid_y) * (self.TILE_W / 2)
        screen_y = self.ORIGIN_Y + (grid_x + grid_y) * (self.TILE_H / 2)
        return screen_x, screen_y

    def _draw_iso_cube(self, pos, color, size_mod=1.0):
        w = self.TILE_W * size_mod
        h = self.TILE_H * size_mod
        d = self.TILE_DEPTH * size_mod
        x, y = int(pos[0]), int(pos[1])
        
        side_color_1 = tuple(max(0, c - 40) for c in color)
        side_color_2 = tuple(max(0, c - 60) for c in color)
        
        top_points = [
            (x, int(y - h / 2)), (int(x + w / 2), y),
            (x, int(y + h / 2)), (int(x - w / 2), y)
        ]
        left_side_points = [
            (int(x - w / 2), y), (x, int(y + h / 2)),
            (x, int(y + h / 2 + d)), (int(x - w / 2), int(y + d))
        ]
        right_side_points = [
            (int(x + w / 2), y), (x, int(y + h / 2)),
            (x, int(y + h / 2 + d)), (int(x + w / 2), int(y + d))
        ]
        
        pygame.gfxdraw.filled_polygon(self.screen, left_side_points, side_color_1)
        pygame.gfxdraw.aapolygon(self.screen, left_side_points, side_color_1)
        pygame.gfxdraw.filled_polygon(self.screen, right_side_points, side_color_2)
        pygame.gfxdraw.aapolygon(self.screen, right_side_points, side_color_2)
        pygame.gfxdraw.filled_polygon(self.screen, top_points, color)
        pygame.gfxdraw.aapolygon(self.screen, top_points, color)

    def _draw_gradient_background(self):
        for y in range(400):
            r = self.COLOR_BG_TOP[0] + (self.COLOR_BG_BOTTOM[0] - self.COLOR_BG_TOP[0]) * y / 400
            g = self.COLOR_BG_TOP[1] + (self.COLOR_BG_BOTTOM[1] - self.COLOR_BG_TOP[1]) * y / 400
            b = self.COLOR_BG_TOP[2] + (self.COLOR_BG_BOTTOM[2] - self.COLOR_BG_TOP[2]) * y / 400
            pygame.draw.line(self.screen, (int(r), int(g), int(b)), (0, y), (640, y))

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    import os
    try:
        os.environ.pop("SDL_VIDEODRIVER")
    except KeyError:
        pass

    env = GameEnv()
    window = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Isometric Hopper")
    
    obs, info = env.reset()
    done = False
    
    print("--- Isometric Hopper ---")
    print(env.game_description)
    print(env.user_guide)
    print("Press 'R' to reset the current stage.")
    
    running = True
    while running:
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                
                if done: # If game is over, only reset is allowed
                    continue

                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    print(f"Action: {action[0]}, Reward: {reward}, Score: {info['score']}, Done: {done}")

        frame = env._get_observation()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        window.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(60)

    pygame.quit()