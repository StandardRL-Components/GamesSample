import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame



class GameEnv(gym.Env):
    """
    An isometric block stacking game. The player's goal is to stack 12 blocks
    on top of a base. A block is always falling, rotating, and drifting.
    The player must press 'space' at the right moment to drop the block.
    If a block is placed off-center, it may cause the tower to collapse.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Press the space bar to drop the falling block. "
        "Time your drop to land the block on the center of the stack."
    )

    game_description = (
        "Stack falling blocks precisely to build a towering structure in this "
        "isometric 2D puzzle game. A steady hand and sharp timing are key!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        # Set environment variable for headless operation if not already set
        if "SDL_VIDEODRIVER" not in os.environ:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Game Constants ---
        self.max_steps = 1000
        self.win_condition = 12
        self.gravity = 0.8

        # --- Color Palette ---
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_GROUND = (60, 65, 70)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)
        self.BLOCK_COLORS = [
            (255, 87, 87),    # Red
            (255, 171, 87),   # Orange
            (255, 255, 87),   # Yellow
            (87, 255, 87),    # Green
            (87, 171, 255),   # Blue
            (171, 87, 255),   # Indigo
            (255, 87, 171),   # Violet
        ]

        # --- Isometric Projection Constants ---
        self.iso_origin = (self.screen_width // 2, self.screen_height // 2 - 50)
        self.iso_angle_cos = math.cos(0.5)
        self.iso_angle_sin = math.sin(0.5)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.stacked_blocks = []
        self.falling_block = None
        self.particles = []
        self.space_was_pressed = False
        self.reward_this_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.space_was_pressed = False

        # Create the base platform
        self.stacked_blocks = [{
            'pos': np.array([0.0, 0.0, -10.0]),
            'size': np.array([120.0, 120.0, 20.0]),
            'rot': 0.0,
            'color': self.COLOR_GROUND,
            'is_base': True
        }]

        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        
        if not self.game_over:
            self._handle_input(action)
            self._update_physics()
            self.reward_this_step += 0.01 # Small reward for surviving
        
        self._update_particles()
        
        self.steps += 1
        
        terminated = self.game_over
        truncated = self.steps >= self.max_steps
        
        if truncated and not self.game_over: # End due to max steps
            self.reward_this_step = -10.0

        reward = self.reward_this_step
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        space_held = action[1] == 1
        
        if space_held and not self.space_was_pressed and self.falling_block:
            self._drop_block()
        
        self.space_was_pressed = space_held

    def _update_physics(self):
        if self.falling_block:
            # Apply gravity and drift
            self.falling_block['pos'] += self.falling_block['vel']
            self.falling_block['vel'][2] -= self.gravity
            
            # Apply rotation
            self.falling_block['rot'] += self.falling_block['rot_vel']
            self.falling_block['rot'] %= 360

            # Check if block has fallen off screen
            if self.falling_block['pos'][2] < -150:
                # FIX: A block falling off-screen is a "miss", not a game over.
                # This prevents termination with no-op actions, which is required
                # for the stability test when auto_advance=True.
                self.reward_this_step -= 2.0  # Penalty for missing
                self._spawn_new_block()

    def _drop_block(self):
        fb = self.falling_block
        if not fb: return
        self.falling_block = None # Block is no longer falling

        # Find the highest block it could land on
        highest_block = max(self.stacked_blocks, key=lambda b: b['pos'][2] + b['size'][2]/2)
        
        # Project falling block's position onto the plane of the highest block's top surface
        landing_z = highest_block['pos'][2] + highest_block['size'][2] / 2
        fb['pos'][2] = landing_z + fb['size'][2] / 2
        
        # Check for stability
        offset = np.linalg.norm(fb['pos'][:2] - highest_block['pos'][:2])
        
        max_offset = (highest_block['size'][0] / 2)
        
        if offset < max_offset:
            # Stable placement
            self.stacked_blocks.append(fb)
            self._spawn_particles(fb['pos'], fb['color'])
            self.reward_this_step += 1.0 # Reward for placing a block
            
            # Check for win condition
            if len(self.stacked_blocks) - 1 >= self.win_condition:
                self.win = True
                self.game_over = True
                self.reward_this_step += 100.0 # Big win reward
            else:
                self._spawn_new_block()
        else:
            # Unstable placement - collapse
            self.particles.append({
                'type': 'block', 'block': fb,
                'vel': np.array([
                    (fb['pos'][0] - highest_block['pos'][0]) * 0.3,
                    (fb['pos'][1] - highest_block['pos'][1]) * 0.3,
                    3.0
                ])
            })
            self._trigger_game_over(collapse=True)

    def _trigger_game_over(self, collapse=False):
        if self.game_over: return
        self.game_over = True
        self.falling_block = None
        self.reward_this_step = -100.0 # Big loss penalty
        if collapse:
            for block in self.stacked_blocks:
                if not block.get('is_base'):
                    self.particles.append({
                        'type': 'block', 'block': block,
                        'vel': self.np_random.uniform(-3, 3, size=3) + np.array([0, 0, 5])
                    })
            self.stacked_blocks = [b for b in self.stacked_blocks if b.get('is_base')]


    def _spawn_new_block(self):
        stack_height = len(self.stacked_blocks) - 1
        color_index = stack_height % len(self.BLOCK_COLORS)
        
        last_block_pos = self.stacked_blocks[-1]['pos']
        spawn_z = last_block_pos[2] + self.stacked_blocks[-1]['size'][2]/2 + 250

        self.falling_block = {
            'pos': np.array([
                self.np_random.uniform(-40, 40),
                self.np_random.uniform(-40, 40),
                spawn_z
            ]),
            'size': np.array([40.0, 40.0, 40.0]),
            'vel': np.array([
                self.np_random.uniform(-0.5, 0.5), # Horizontal drift
                self.np_random.uniform(-0.5, 0.5), # Horizontal drift
                0.0
            ]),
            'rot': self.np_random.uniform(0, 360),
            'rot_vel': self.np_random.uniform(-1.5, 1.5),
            'color': self.BLOCK_COLORS[color_index]
        }

    def _spawn_particles(self, pos, color):
        for _ in range(20):
            vel = self.np_random.uniform(-2, 2, size=3)
            vel[2] = abs(vel[2]) + 1 # Force upwards
            self.particles.append({
                'type': 'spark',
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(20, 41),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            if p['type'] == 'spark':
                p['pos'] += p['vel']
                p['vel'][2] -= 0.2 # Gravity on particles
                p['life'] -= 1
            elif p['type'] == 'block':
                p['block']['pos'] += p['vel']
                p['block']['rot'] += 10 # Tumble
                p['vel'][2] -= self.gravity
        
        self.particles = [p for p in self.particles if p.get('life', 1) > 0 and p.get('block', {}).get('pos', [0,0,0])[2] > -150]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "blocks_stacked": max(0, len(self.stacked_blocks) - 1),
        }

    def _render_grid(self):
        for i in range(-15, 16):
            p1 = self._iso_transform(-200, i * 20, 0)
            p2 = self._iso_transform(200, i * 20, 0)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
            p1 = self._iso_transform(i * 20, -200, 0)
            p2 = self._iso_transform(i * 20, 200, 0)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)

    def _render_game(self):
        render_list = self.stacked_blocks[:]
        if self.falling_block:
            render_list.append(self.falling_block)
        
        render_list.sort(key=lambda b: b['pos'][0] + b['pos'][1] - b['pos'][2])

        for block in render_list:
            self._draw_iso_cube(block['pos'], block['size'], block['rot'], block['color'])
            
        for p in self.particles:
            if p['type'] == 'spark':
                size = max(0, p['life'] / 10.0)
                pos_2d = self._iso_transform(p['pos'][0], p['pos'][1], p['pos'][2])
                pygame.draw.circle(self.screen, p['color'], pos_2d, size)
            elif p['type'] == 'block':
                b = p['block']
                self._draw_iso_cube(b['pos'], b['size'], b['rot'], b['color'])

    def _render_ui(self):
        block_count = max(0, len(self.stacked_blocks) - 1)
        score_text = f"BLOCKS: {block_count}/{self.win_condition}"
        self._draw_text(score_text, (20, 20), self.font_ui)
        
        if self.game_over:
            if self.win:
                msg = "TOWER COMPLETE!"
                color = (100, 255, 100)
            else:
                msg = "TOWER COLLAPSED"
                color = (255, 100, 100)
            
            self._draw_text(msg, (self.screen_width/2, self.screen_height/2), self.font_game_over, center=True, color=color)

    def _draw_text(self, text, pos, font, color=None, center=False):
        if color is None: color = self.COLOR_TEXT
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_surf = font.render(text, True, color)
        
        if center:
            text_rect = text_surf.get_rect(center=pos)
            shadow_rect = shadow_surf.get_rect(center=(pos[0]+2, pos[1]+2))
        else:
            text_rect = text_surf.get_rect(topleft=pos)
            shadow_rect = shadow_surf.get_rect(topleft=(pos[0]+2, pos[1]+2))
            
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _iso_transform(self, x, y, z):
        screen_x = self.iso_origin[0] + (x - y) * self.iso_angle_cos
        screen_y = self.iso_origin[1] + (x + y) * self.iso_angle_sin - z
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, pos_3d, size_3d, rotation, color):
        w, d, h = size_3d[0] / 2, size_3d[1] / 2, size_3d[2]
        x, y, z = pos_3d[0], pos_3d[1], pos_3d[2] - h/2
        
        corners = [
            np.array([-w, -d]), np.array([w, -d]), np.array([w, d]), np.array([-w, d])
        ]
        
        rad = math.radians(rotation)
        rot_matrix = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])
        rotated_corners = [rot_matrix @ c for c in corners]
        
        p = [self._iso_transform(c[0] + x, c[1] + y, z) for c in rotated_corners]
        p_top = [self._iso_transform(c[0] + x, c[1] + y, z + h) for c in rotated_corners]

        top_face = [p_top[0], p_top[1], p_top[2], p_top[3]]
        side_face_1 = [p[0], p[1], p_top[1], p_top[0]]
        side_face_2 = [p[1], p[2], p_top[2], p_top[1]]

        top_color = color
        side_color_1 = tuple(int(c * 0.8) for c in color)
        side_color_2 = tuple(int(c * 0.6) for c in color)
        
        pygame.gfxdraw.filled_polygon(self.screen, side_face_1, side_color_1)
        pygame.gfxdraw.aapolygon(self.screen, side_face_1, side_color_1)
        
        pygame.gfxdraw.filled_polygon(self.screen, side_face_2, side_color_2)
        pygame.gfxdraw.aapolygon(self.screen, side_face_2, side_color_2)
        
        pygame.gfxdraw.filled_polygon(self.screen, top_face, top_color)
        pygame.gfxdraw.aapolygon(self.screen, top_face, top_color)
        
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    env.validate_implementation()
    
    # --- Test Reset ---
    print("\n--- Testing Reset ---")
    obs, info = env.reset()
    print("Reset successful.")
    print("Initial Info:", info)
    assert obs.shape == (400, 640, 3)
    assert isinstance(info, dict)

    # --- Test a few Steps ---
    print("\n--- Testing Steps ---")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Reward={reward:.2f}, Terminated={terminated}, Truncated={truncated}, Info={info}")
        if terminated or truncated:
            print("Episode ended. Resetting.")
            env.reset()
    
    # --- Test a specific action (dropping a block) ---
    print("\n--- Testing specific drop action ---")
    env.reset()
    # Let the block fall for a bit
    for _ in range(50):
        env.step([0, 0, 0]) # no-op
    # Now, drop the block
    print("Dropping block...")
    obs, reward, terminated, truncated, info = env.step([0, 1, 0])
    print(f"Drop Action: Reward={reward:.2f}, Terminated={terminated}, Truncated={truncated}, Info={info}")
    
    env.close()
    print("\nEnvironment tests passed.")