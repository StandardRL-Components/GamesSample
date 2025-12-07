import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:18:02.033427
# Source Brief: brief_01008.md
# Brief Index: 1008
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Stack blocks of different materials to build the tallest, most stable tower possible. "
        "Race against the clock and prevent your structure from collapsing."
    )
    user_guide = (
        "Controls: Use ←→ to move the block. The material changes based on key presses: "
        "hold space for wood, shift for metal, or neither for rubber."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- CRITICAL: Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FLOOR_Y = self.HEIGHT - 40
        self.MAX_STEPS = 30 * 60 # 30 seconds at 60fps
        self.WIN_STABILITY_DURATION = 10 * 60 # 10 seconds

        # --- Visuals ---
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_FLOOR = (60, 65, 70)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_STABILITY_GOOD = (0, 255, 127)
        self.COLOR_STABILITY_BAD = (255, 69, 0)
        
        self.MATERIALS = {
            "wood": {"color": (160, 110, 70), "density": 1.0, "friction": 0.7, "restitution": 0.1},
            "metal": {"color": (192, 192, 208), "density": 2.5, "friction": 0.2, "restitution": 0.8},
            "rubber": {"color": (70, 80, 160), "density": 0.8, "friction": 0.9, "restitution": 0.05},
        }

        # --- Physics ---
        self.GRAVITY = 0.05
        self.PLAYER_SPEED = 4
        self.BLOCK_WIDTH, self.BLOCK_HEIGHT = 80, 25
        self.TILT_THRESHOLD = math.pi / 4  # 45 degrees
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_huge = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- State variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_left = 0
        self.stability_timer = 0
        self.placed_blocks = []
        self.current_block = None
        self.next_block_material = "wood"
        self.fall_speed_multiplier = 1.0
        self.placed_count = 0
        self.particles = []
        self.event_reward = 0.0
        self.last_total_tilt = 0.0
        
        # In Gymnasium, the random number generator is part of the environment
        # and should be seeded via reset().
        self.np_random = None

    def _create_block(self, pos, material_name):
        return {
            "pos": list(pos),
            "size": (self.BLOCK_WIDTH, self.BLOCK_HEIGHT),
            "material": material_name,
            "angle": 0.0,
            "angular_velocity": 0.0,
            "static": False
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_left = self.MAX_STEPS
        self.stability_timer = 0
        self.placed_count = 0
        self.fall_speed_multiplier = 1.0
        
        # Create a static floor block
        self.placed_blocks = [{
            "pos": [self.WIDTH / 2, self.FLOOR_Y + self.BLOCK_HEIGHT/2],
            "size": (self.WIDTH, self.BLOCK_HEIGHT),
            "material": "metal",
            "angle": 0.0,
            "angular_velocity": 0.0,
            "static": True
        }]
        
        self.particles = []
        self.event_reward = 0.0
        self.last_total_tilt = 0.0

        self._spawn_new_block()
        self._spawn_new_block() # Call twice to populate current and next

        return self._get_observation(), self._get_info()

    def _spawn_new_block(self):
        self.current_block = self._create_block(
            pos=[self.WIDTH / 2, self.BLOCK_HEIGHT],
            material_name=self.next_block_material
        )
        self.next_block_material = self.np_random.choice(list(self.MATERIALS.keys()))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_left -= 1
        self.event_reward = 0.0

        self._handle_input(action)
        self._update_physics()
        self._update_game_state()
        
        reward = self._calculate_reward()
        self.score += reward
        
        terminated = self.game_over
        truncated = False # No truncation condition in this game besides termination

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Horizontal movement
        if movement == 3: # Left
            self.current_block['pos'][0] -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.current_block['pos'][0] += self.PLAYER_SPEED

        # Clamp position
        half_w = self.current_block['size'][0] / 2
        self.current_block['pos'][0] = max(half_w, min(self.WIDTH - half_w, self.current_block['pos'][0]))

        # Material selection
        if shift_held:
            self.current_block['material'] = 'metal'
        elif space_held:
            self.current_block['material'] = 'wood'
        else:
            self.current_block['material'] = 'rubber'

    def _update_physics(self):
        # Move current block down
        self.current_block['pos'][1] += self.GRAVITY * 15 * self.fall_speed_multiplier

        # Check for collision
        collided = False
        for other_block in reversed(self.placed_blocks):
            if self._check_collision(self.current_block, other_block):
                collided = True
                # Nudge block to rest on top
                self.current_block['pos'][1] = other_block['pos'][1] - other_block['size'][1]/2 - self.current_block['size'][1]/2
                break
        
        if collided:
            # --- Sound effect: Block place ---
            self.placed_blocks.append(self.current_block)
            self._create_impact_particles(self.current_block['pos'], self.current_block['material'])
            self.event_reward += 1.0 # Reward for placing a block
            self.placed_count += 1
            if self.placed_count % 5 == 0:
                self.fall_speed_multiplier += 0.05
            self._spawn_new_block()
        
        # Simulate stack physics
        self._simulate_stack()
        self._update_particles()

    def _get_support_points(self, block_idx):
        block = self.placed_blocks[block_idx]
        min_x, max_x = float('inf'), float('-inf')
        supported = False
        
        for i in range(block_idx):
            support_block = self.placed_blocks[i]
            if self._check_collision(block, support_block):
                # A simple approximation of support surface
                w, _ = support_block['size']
                px, _ = support_block['pos']
                angle = support_block['angle']
                
                # Get corners of the support block
                p1x = px - w/2 * math.cos(angle)
                p2x = px + w/2 * math.cos(angle)

                min_x = min(min_x, p1x, p2x)
                max_x = max(max_x, p1x, p2x)
                supported = True

        return min_x, max_x, supported

    def _simulate_stack(self):
        collapse_indices = set()
        
        # Iterate multiple times for stability
        for _ in range(5):
            for i in range(1, len(self.placed_blocks)):
                block = self.placed_blocks[i]
                
                # Simple gravity pull if not on floor
                block['pos'][1] += self.GRAVITY * 2

                # Check for collisions and resolve
                for j in range(i):
                    other = self.placed_blocks[j]
                    if self._check_collision(block, other):
                         block['pos'][1] = other['pos'][1] - other['size'][1]/2 - block['size'][1]/2

                # Stability calculation
                min_sup_x, max_sup_x, is_supported = self._get_support_points(i)
                
                if not is_supported and block['pos'][1] < self.FLOOR_Y - block['size'][1]:
                    continue # Free falling

                com_x = block['pos'][0] - block['size'][1]/2 * math.sin(block['angle'])
                
                torque = 0
                if com_x < min_sup_x:
                    torque = (com_x - min_sup_x) * 0.001
                elif com_x > max_sup_x:
                    torque = (com_x - max_sup_x) * 0.001
                
                density = self.MATERIALS[block['material']]['density']
                block['angular_velocity'] += torque * density

                # Damping based on friction
                friction = self.MATERIALS[block['material']]['friction']
                block['angular_velocity'] *= (0.95 - (friction * 0.05))
                block['angle'] += block['angular_velocity']

                if abs(block['angle']) > self.TILT_THRESHOLD:
                    collapse_indices.add(i)

        if collapse_indices:
            # --- Sound effect: Collapse ---
            self.event_reward -= 5.0
            
            # Chain reaction: remove everything above collapsed blocks
            final_collapse_set = set()
            for idx in sorted(list(collapse_indices), reverse=True):
                if idx not in final_collapse_set:
                    final_collapse_set.add(idx)
                    # Also mark everything that was on top of it
                    for k in range(idx + 1, len(self.placed_blocks)):
                        final_collapse_set.add(k)
            
            # Create particles for each collapsed block before removing
            for idx in sorted(list(final_collapse_set), reverse=True):
                b = self.placed_blocks[idx]
                self._create_impact_particles(b['pos'], b['material'], 20)

            self.placed_blocks = [b for i, b in enumerate(self.placed_blocks) if i not in final_collapse_set]
            
            if len(self.placed_blocks) <= 1: # Only floor left
                self.game_over = True
                self.event_reward -= 100 # Penalty for total collapse

    def _check_collision(self, block1, block2):
        # Simple AABB collision check for this game's purpose
        rect1 = self._get_rotated_rect(block1)
        rect2 = self._get_rotated_rect(block2)
        return rect1.colliderect(rect2)
    
    def _get_rotated_rect(self, block):
        # Pygame's Rects don't handle rotation, so we approximate
        w, h = block['size']
        angle = block['angle']
        abs_cos = abs(math.cos(angle))
        abs_sin = abs(math.sin(angle))
        new_w = w * abs_cos + h * abs_sin
        new_h = w * abs_sin + h * abs_cos
        return pygame.Rect(block['pos'][0] - new_w/2, block['pos'][1] - new_h/2, new_w, new_h)

    def _update_game_state(self):
        total_tilt = sum(abs(b['angle']) for b in self.placed_blocks if not b['static'])
        self.last_total_tilt = total_tilt

        if total_tilt < 0.1: # Stability threshold in radians
            self.stability_timer += 1
        else:
            self.stability_timer = 0

        if self.stability_timer >= self.WIN_STABILITY_DURATION:
            self.game_over = True
            # --- Sound effect: Win ---
            self.event_reward += 100.0
        
        if self.time_left <= 0:
            self.game_over = True
            if self.stability_timer < self.WIN_STABILITY_DURATION:
                # --- Sound effect: Lose/Timeout ---
                self.event_reward -= 100.0
    
    def _calculate_reward(self):
        # Continuous reward for stability
        tilt_penalty = -0.5 * self.last_total_tilt
        stability_reward = 0.1 if self.stability_timer > 0 else 0.0

        total_reward = self.event_reward + tilt_penalty + stability_reward
        return np.clip(total_reward, -10.0, 10.0) if not self.game_over else self.event_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

    def _render_game(self):
        # Draw all placed blocks
        for block in self.placed_blocks:
            self._render_block(block)
        
        # Draw current falling block
        if self.current_block:
            self._render_block(self.current_block)
        
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['size']))

    def _render_block(self, block):
        w, h = block['size']
        x, y = block['pos']
        material_props = self.MATERIALS.get(block['material'], self.MATERIALS['wood'])
        color = material_props['color']
        
        # Create a surface for the block to handle rotation
        block_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        
        # Base color
        block_surf.fill(color)

        # Texture
        if block['material'] == 'wood':
            for i in range(5):
                pygame.draw.line(block_surf, (0,0,0,30), (self.np_random.integers(0,w),0), (self.np_random.integers(0,w),h), 2)
        elif block['material'] == 'metal':
            pygame.draw.line(block_surf, (255,255,255,50), (0,5), (w,5), 3)
            pygame.draw.line(block_surf, (0,0,0,20), (0,h-5), (w,h-5), 2)
        elif block['material'] == 'rubber':
            pygame.draw.circle(block_surf, (0,0,0,20), (w/4, h/2), 5)
            pygame.draw.circle(block_surf, (0,0,0,20), (w*3/4, h/2), 5)

        # Rotate and blit
        rotated_surf = pygame.transform.rotate(block_surf, math.degrees(block['angle']))
        new_rect = rotated_surf.get_rect(center=(int(x), int(y)))
        self.screen.blit(rotated_surf, new_rect.topleft)

    def _render_ui(self):
        # Timer
        time_ratio = self.time_left / self.MAX_STEPS
        time_color = (int(255 * (1 - time_ratio)), int(255 * time_ratio), 0)
        time_text = f"Time: {self.time_left / 60:.1f}s"
        rendered_text = self.font_large.render(time_text, True, time_color)
        self.screen.blit(rendered_text, (self.WIDTH - rendered_text.get_width() - 10, 10))
        
        # Stability Bar
        stability_ratio = min(1.0, self.stability_timer / self.WIN_STABILITY_DURATION)
        bar_width = 200
        bar_height = 20
        fill_width = int(bar_width * stability_ratio)
        pygame.draw.rect(self.screen, self.COLOR_GRID, (10, 10, bar_width, bar_height))
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_STABILITY_GOOD, (10, 10, fill_width, bar_height))
        stability_text = self.font_small.render("STABILITY", True, self.COLOR_UI_TEXT)
        self.screen.blit(stability_text, (15, 13))

        # Next Block Preview
        preview_text = self.font_small.render("Next:", True, self.COLOR_UI_TEXT)
        self.screen.blit(preview_text, (self.WIDTH/2 - 80, self.HEIGHT - 35))
        preview_block = self._create_block((self.WIDTH/2 + 20, self.HEIGHT - 22), self.next_block_material)
        preview_block['size'] = (self.BLOCK_WIDTH * 0.7, self.BLOCK_HEIGHT * 0.7)
        self._render_block(preview_block)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            won = self.stability_timer >= self.WIN_STABILITY_DURATION
            msg = "YOU WIN!" if won else "GAME OVER"
            color = self.COLOR_STABILITY_GOOD if won else self.COLOR_STABILITY_BAD
            
            rendered_text = self.font_huge.render(msg, True, color)
            self.screen.blit(rendered_text, (self.WIDTH/2 - rendered_text.get_width()/2, self.HEIGHT/2 - rendered_text.get_height()/2))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "stability_timer": self.stability_timer,
            "placed_blocks": len(self.placed_blocks) -1
        }
    
    def _create_impact_particles(self, pos, material, count=10):
        color = self.MATERIALS[material]['color']
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'size': self.np_random.uniform(2, 5),
                'life': self.np_random.integers(20, 41),
                'color': color
            })
    
    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += self.GRAVITY * 0.5 # Particles have gravity
            p['life'] -= 1
            p['size'] -= 0.05
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0]

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment.
    os.environ.pop("SDL_VIDEODRIVER", None)
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Stacker Environment")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        # Display score for manual play
        score_text = env.font_small.render(f"Total Reward: {total_reward:.2f}", True, (255, 255, 255))
        screen.blit(score_text, (10, env.HEIGHT - 25))

        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)

        clock.tick(60) # Run at 60 FPS
        
    env.close()