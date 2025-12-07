import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:42:54.031651
# Source Brief: brief_00586.md
# Brief Index: 586
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: A Gymnasium environment where the agent builds a tower block by block.

    The goal is to build the tallest, most stable structure possible. The game ends
    if the structure collapses or a step limit is reached. Visuals are a key focus,
    with an isometric perspective, particle effects, and clear UI feedback.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right) for the placement cursor.
    - actions[1]: Place Block (0=released, 1=pressed).
    - actions[2]: Cycle Block Type (0=released, 1=pressed).

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB array representing the game screen.

    Reward Structure:
    - +0.1 for each successfully placed block.
    - +1.0 for reaching a new maximum height.
    - -0.5 for placing a block that significantly reduces stability.
    - +50 for reaching a height of 50, +100 for 100.
    - -50 for a structure collapse.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Build a tower block by block. The goal is to build the tallest, most stable structure possible."
    user_guide = "Controls: Use arrow keys ←→↑↓ to move the cursor. Press space to place a block and shift to cycle block types."
    auto_advance = False

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 12
    GRID_DEPTH = 12
    MAX_STEPS = 1000
    
    # Visuals
    COLOR_BG = (20, 30, 40)
    COLOR_GROUND_LIGHT = (60, 70, 80)
    COLOR_GROUND_DARK = (50, 60, 70)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_STABILITY_GOOD = (60, 220, 120)
    COLOR_STABILITY_MID = (240, 200, 80)
    COLOR_STABILITY_BAD = (250, 80, 80)
    
    BLOCK_PALETTE = [
        ((80, 160, 255), (60, 120, 225), (40, 80, 195)),   # Blue
        ((255, 120, 80), (225, 90, 60), (195, 60, 40)),    # Orange
        ((100, 220, 100), (80, 190, 80), (60, 160, 60)),   # Green
        ((240, 80, 240), (210, 60, 210), (180, 40, 180)),  # Magenta
        ((240, 240, 80), (210, 210, 60), (180, 180, 40)),  # Yellow
    ]

    # Isometric projection parameters
    BLOCK_ISO_WIDTH = 32
    BLOCK_ISO_HEIGHT = 16
    BLOCK_RENDER_HEIGHT = 24
    
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
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = self.SCREEN_HEIGHT - 50

        # These attributes are defined here but initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.blocks = []
        self.block_positions = set()
        self.cursor_pos = [0, 0] # Grid (x, z)
        self.selected_block_type = 0
        self.max_height = 0
        self.current_height = 0
        self.stability = 1.0
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.collapse_info = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.blocks = []
        self.block_positions = set()
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_DEPTH // 2]
        self.selected_block_type = 0
        self.max_height = 0
        self.current_height = 0
        self.stability = 1.0
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.collapse_info = None
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If game is over, only process collapse animation and step counter
            self.steps += 1
            terminated = (self.steps >= self.MAX_STEPS) or (self.collapse_info and self.collapse_info['timer'] <= 0)
            return self._get_observation(), 0, terminated, False, self._get_info()

        self.steps += 1
        reward = 0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        # --- Handle Actions ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up (decr z)
        elif movement == 2: self.cursor_pos[1] += 1 # Down (incr z)
        elif movement == 3: self.cursor_pos[0] -= 1 # Left (decr x)
        elif movement == 4: self.cursor_pos[0] += 1 # Right (incr x)
        
        # Wrap cursor around grid
        self.cursor_pos[0] %= self.GRID_WIDTH
        self.cursor_pos[1] %= self.GRID_DEPTH

        if shift_pressed:
            self.selected_block_type = (self.selected_block_type + 1) % len(self.BLOCK_PALETTE)
            # sfx: cycle_weapon.wav

        if space_pressed:
            reward += self._place_block()
            # sfx: place_block.wav
        
        # --- Update Game State ---
        self._update_particles()
        
        prev_stability = self.stability
        self._update_state_metrics()
        if self.stability < prev_stability - 0.2:
            reward -= 0.5

        if self._check_for_collapse():
            self.game_over = True
            reward -= 50
            # sfx: structure_collapse.wav

        # --- Finalize Step ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = False
        self.score += reward
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _place_block(self):
        cx, cz = self.cursor_pos
        
        # Find highest block at this (x, z) location to stack on top of
        placement_y = 0
        for block in self.blocks:
            if block['gx'] == cx and block['gz'] == cz:
                placement_y = max(placement_y, block['gy'] + 1)
        
        # Check if placement is valid (not inside another block)
        if (cx, placement_y, cz) in self.block_positions:
            return 0 # Invalid placement

        new_block = {'gx': cx, 'gy': placement_y, 'gz': cz, 'type': self.selected_block_type, 'stable': True}
        self.blocks.append(new_block)
        self.block_positions.add((cx, placement_y, cz))
        
        # Particle effect on placement
        px, py = self._iso_transform(cx, placement_y, cz)
        self._spawn_particles((px, py), self.BLOCK_PALETTE[self.selected_block_type][0], 20, 2)
        
        reward = 0.1

        # Random block removal (20% chance)
        if self.np_random.random() < 0.2:
            removable_blocks = [b for i, b in enumerate(self.blocks) if b['gy'] > 0 and b != new_block]
            if removable_blocks:
                block_to_remove = random.choice(removable_blocks)
                self.blocks.remove(block_to_remove)
                self.block_positions.remove((block_to_remove['gx'], block_to_remove['gy'], block_to_remove['gz']))
                
                # Particle effect on removal
                rpx, rpy = self._iso_transform(block_to_remove['gx'], block_to_remove['gy'], block_to_remove['gz'])
                self._spawn_particles((rpx, rpy), (200,200,200), 15, 3)
                # sfx: block_removed.wav

        # Update height and check for reward
        new_height = 0
        if self.blocks:
            new_height = max(b['gy'] for b in self.blocks) + 1
        
        if new_height > self.max_height:
            reward += 1.0
            self.max_height = new_height
            if self.max_height == 50: reward += 50
            if self.max_height == 100: reward += 100
        
        return reward

    def _check_for_collapse(self):
        if not self.blocks:
            return False

        q = [b for b in self.blocks if b['gy'] == 0]
        supported_blocks = set((b['gx'], b['gy'], b['gz']) for b in q)
        
        i = 0
        while i < len(q):
            parent = q[i]
            i += 1
            
            child_pos = (parent['gx'], parent['gy'] + 1, parent['gz'])
            if child_pos in self.block_positions and child_pos not in supported_blocks:
                child_block = next(b for b in self.blocks if (b['gx'], b['gy'], b['gz']) == child_pos)
                q.append(child_block)
                supported_blocks.add(child_pos)

        if len(supported_blocks) < len(self.blocks):
            unsupported = [b for b in self.blocks if (b['gx'], b['gy'], b['gz']) not in supported_blocks]
            self.collapse_info = {'timer': 60, 'unsupported': unsupported}
            return True
        return False

    def _update_state_metrics(self):
        if not self.blocks:
            self.current_height = 0
            self.stability = 1.0
            return

        self.current_height = max(b['gy'] for b in self.blocks) + 1

        # Stability based on Center of Mass vs. Base Support
        base_blocks = [b for b in self.blocks if b['gy'] == 0]
        if not base_blocks:
            self.stability = 0
            return

        sum_x, sum_z, total_mass = 0, 0, 0
        for block in self.blocks:
            mass = block['gy'] + 1 # Higher blocks contribute more to instability
            sum_x += block['gx'] * mass
            sum_z += block['gz'] * mass
            total_mass += mass
        
        if total_mass == 0:
            self.stability = 1.0
            return

        com_x = sum_x / total_mass
        com_z = sum_z / total_mass

        min_bx = min(b['gx'] for b in base_blocks)
        max_bx = max(b['gx'] for b in base_blocks)
        min_bz = min(b['gz'] for b in base_blocks)
        max_bz = max(b['gz'] for b in base_blocks)
        
        center_bx = (min_bx + max_bx) / 2
        center_bz = (min_bz + max_bz) / 2
        width_b = (max_bx - min_bx) + 1
        depth_b = (max_bz - min_bz) + 1

        dist_x = abs(com_x - center_bx)
        dist_z = abs(com_z - center_bz)

        # Stability is 1 if CoM is over the base, decreases to 0 at 2x the base width
        stability_x = max(0, 1 - (dist_x / width_b))
        stability_z = max(0, 1 - (dist_z / depth_b))
        self.stability = min(stability_x, stability_z)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "height": self.current_height, "stability": self.stability}

    # --- Rendering Methods ---
    
    def _iso_transform(self, x, y, z):
        screen_x = self.origin_x + (x - z) * self.BLOCK_ISO_WIDTH
        screen_y = self.origin_y + (x + z) * self.BLOCK_ISO_HEIGHT - y * self.BLOCK_RENDER_HEIGHT
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, pos, color_palette, wireframe=False, alpha=255):
        x, y = pos
        top_face = [
            (x, y - self.BLOCK_ISO_HEIGHT),
            (x + self.BLOCK_ISO_WIDTH, y),
            (x, y + self.BLOCK_ISO_HEIGHT),
            (x - self.BLOCK_ISO_WIDTH, y)
        ]
        
        if wireframe:
            pygame.draw.polygon(surface, color_palette[0], top_face, 2)
            # Draw connecting lines for wireframe
            for i in range(4):
                p1 = top_face[i]
                p2 = (top_face[i][0], top_face[i][1] + self.BLOCK_RENDER_HEIGHT)
                pygame.draw.line(surface, color_palette[0], p1, p2, 2)
        else:
            # Draw faces with 3D shading
            left_face = [top_face[3], top_face[2], (top_face[2][0], top_face[2][1] + self.BLOCK_RENDER_HEIGHT), (top_face[3][0], top_face[3][1] + self.BLOCK_RENDER_HEIGHT)]
            right_face = [top_face[2], top_face[1], (top_face[1][0], top_face[1][1] + self.BLOCK_RENDER_HEIGHT), (top_face[2][0], top_face[2][1] + self.BLOCK_RENDER_HEIGHT)]
            
            pygame.draw.polygon(surface, color_palette[2], left_face) # Darker side
            pygame.draw.polygon(surface, color_palette[1], right_face) # Mid side
            pygame.draw.polygon(surface, color_palette[0], top_face) # Bright top
            
            if alpha < 255:
                s = pygame.Surface((self.BLOCK_ISO_WIDTH*2, self.BLOCK_ISO_HEIGHT*2 + self.BLOCK_RENDER_HEIGHT), pygame.SRCALPHA)
                s.set_alpha(alpha)
                pygame.draw.polygon(s, (255, 0, 0), [(p[0]-x+self.BLOCK_ISO_WIDTH, p[1]-y+self.BLOCK_ISO_HEIGHT) for p in left_face])
                pygame.draw.polygon(s, (255, 0, 0), [(p[0]-x+self.BLOCK_ISO_WIDTH, p[1]-y+self.BLOCK_ISO_HEIGHT) for p in right_face])
                pygame.draw.polygon(s, (255, 0, 0), [(p[0]-x+self.BLOCK_ISO_WIDTH, p[1]-y+self.BLOCK_ISO_HEIGHT) for p in top_face])
                surface.blit(s, (x-self.BLOCK_ISO_WIDTH, y-self.BLOCK_ISO_HEIGHT))

    def _render_game(self):
        # Draw ground
        for x in range(self.GRID_WIDTH):
            for z in range(self.GRID_DEPTH):
                px, py = self._iso_transform(x, -0.5, z)
                color = self.COLOR_GROUND_LIGHT if (x + z) % 2 == 0 else self.COLOR_GROUND_DARK
                self._draw_iso_cube(self.screen, (px, py), [color, color, color])

        # Handle collapse animation
        if self.collapse_info:
            self.collapse_info['timer'] -= 1
            for b in self.collapse_info['unsupported']:
                b['gy'] -= 0.2 # Fall down
                b['gx'] += self.np_random.uniform(-0.1, 0.1) # Jitter
                b['gz'] += self.np_random.uniform(-0.1, 0.1)

        # Sort blocks for painter's algorithm
        sorted_blocks = sorted(self.blocks, key=lambda b: (b['gy'], b['gx'] + b['gz']))

        # Draw blocks
        for block in sorted_blocks:
            px, py = self._iso_transform(block['gx'], block['gy'], block['gz'])
            self._draw_iso_cube(self.screen, (px, py), self.BLOCK_PALETTE[block['type']])

        # Draw placement cursor
        if not self.game_over:
            cx, cz = self.cursor_pos
            placement_y = 0
            for block in self.blocks:
                if block['gx'] == cx and block['gz'] == cz:
                    placement_y = max(placement_y, block['gy'] + 1)
            
            cursor_px, cursor_py = self._iso_transform(cx, placement_y, cz)
            self._draw_iso_cube(self.screen, (cursor_px, cursor_py), self.BLOCK_PALETTE[self.selected_block_type], wireframe=True)

        # Draw particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            
            alpha = max(0, min(255, p['life'] * 5))
            radius = p['radius'] * (p['life'] / p['initial_life'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(radius), (*p['color'], alpha))

    def _render_ui(self):
        # Height Display
        height_text = self.font_large.render(f"Height: {self.current_height}", True, self.COLOR_UI_TEXT)
        self.screen.blit(height_text, (10, 10))
        
        # Stability Bar
        stability_text = self.font_large.render("Stability", True, self.COLOR_UI_TEXT)
        self.screen.blit(stability_text, (self.SCREEN_WIDTH - 160, 10))
        bar_x, bar_y, bar_w, bar_h = self.SCREEN_WIDTH - 160, 40, 150, 20
        pygame.draw.rect(self.screen, self.COLOR_GROUND_DARK, (bar_x, bar_y, bar_w, bar_h))
        
        stability_color = self.COLOR_STABILITY_BAD
        if self.stability > 0.75: stability_color = self.COLOR_STABILITY_GOOD
        elif self.stability > 0.25: stability_color = self.COLOR_STABILITY_MID
        
        fill_w = max(0, self.stability * bar_w)
        pygame.draw.rect(self.screen, stability_color, (bar_x, bar_y, fill_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_w, bar_h), 2)
        
        # Game Over Text
        if self.game_over and self.collapse_info:
            text = self.font_large.render("STRUCTURE COLLAPSED", True, self.COLOR_STABILITY_BAD)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            pygame.draw.rect(self.screen, (0,0,0,150), text_rect.inflate(20, 10))
            self.screen.blit(text, text_rect)

    # --- Utility Methods ---
    
    def _spawn_particles(self, pos, color, count, speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = [math.cos(angle) * self.np_random.uniform(0.5, 1) * speed, 
                   math.sin(angle) * self.np_random.uniform(0.5, 1) * speed - speed*0.5]
            life = self.np_random.integers(30, 60)
            self.particles.append({
                'pos': list(pos), 'vel': vel, 'color': color, 
                'life': life, 'initial_life': life, 'radius': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Use a display separate from the environment's internal surface
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Builder")
    clock = pygame.time.Clock()

    action = [0, 0, 0] # [movement, space, shift]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                # Set held state
                if event.key == pygame.K_SPACE: action[1] = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                if event.key == pygame.K_UP: action[0] = 1
                if event.key == pygame.K_DOWN: action[0] = 2
                if event.key == pygame.K_LEFT: action[0] = 3
                if event.key == pygame.K_RIGHT: action[0] = 4
        
        if terminated:
            # On termination, allow reset with any key
            keys = pygame.key.get_pressed()
            if any(keys):
                obs, info = env.reset()
                terminated = False
                action = [0, 0, 0]
        else:
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Height: {info['height']}, Stability: {info['stability']:.2f}")

        # Reset non-movement actions after one step to simulate a press
        action[0] = 0
        action[1] = 0
        action[2] = 0
        
        # Manually get key states for continuous movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()