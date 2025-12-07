import gymnasium as gym
import os
import pygame
import numpy as np
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()
pygame.font.init()

# The following import is conditional to handle different pygame versions
try:
    import pygame.gfxdraw
except ImportError:
    # In some newer pygame versions, gfxdraw might not be a separate module
    # but its functions are available directly under pygame.
    # We create a dummy object that redirects calls.
    class GfxDrawRedirect:
        def __getattr__(self, name):
            if hasattr(pygame, name):
                return getattr(pygame, name)
            raise AttributeError(f"'pygame' has no attribute '{name}', and 'pygame.gfxdraw' could not be imported.")
    pygame.gfxdraw = GfxDrawRedirect()


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Stack falling blocks to reach the target height before time runs out. "
        "Align blocks perfectly to trigger chain reactions for bonus points."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to move the falling block. "
        "Press space to drop it instantly."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.GROUND_Y = self.SCREEN_HEIGHT - 20

        # Game parameters
        self.TARGET_HEIGHT = 100
        self.MAX_TIME_SECONDS = 45
        self.MAX_STEPS = self.MAX_TIME_SECONDS * self.FPS
        self.PLAYER_MOVE_SPEED = 8
        self.PERFECT_ALIGN_TOLERANCE = 4

        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (5, 10, 20)
        self.COLOR_GROUND = (100, 110, 130)
        self.COLOR_UI_TEXT = (230, 230, 255)
        self.COLOR_TARGET_LINE = (255, 50, 50, 150)
        self.COLOR_TIMER_BAR = (50, 200, 255)
        self.COLOR_TIMER_BAR_BG = (50, 80, 120)
        self.BLOCK_COLORS = [
            (255, 80, 80), (80, 255, 80), (80, 80, 255),
            (255, 255, 80), (80, 255, 255), (255, 80, 255)
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        self.stack_height = 0
        self.stacked_blocks = []
        self.falling_block = None
        self.fall_speed = 0.0
        self.obstacles = []
        self.particles = []
        self.last_space_held = False
        self.color_index = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_TIME_SECONDS * self.FPS
        self.stack_height = 0
        self.stacked_blocks = []
        self.obstacles = []
        self.particles = []
        self.last_space_held = False
        self.fall_speed = 1.5
        self.color_index = self.np_random.integers(0, len(self.BLOCK_COLORS))

        self._spawn_falling_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        
        # --- Action Handling ---
        movement = action[0]
        space_held = action[1] == 1

        # Movement
        if self.falling_block:
            if movement == 3:  # Left
                self.falling_block['rect'].x -= self.PLAYER_MOVE_SPEED
            elif movement == 4:  # Right
                self.falling_block['rect'].x += self.PLAYER_MOVE_SPEED

            # Clamp falling block position
            self.falling_block['rect'].left = max(0, self.falling_block['rect'].left)
            self.falling_block['rect'].right = min(self.SCREEN_WIDTH, self.falling_block['rect'].right)

        # Place block (triggered on press)
        space_pressed = space_held and not self.last_space_held
        if space_pressed and self.falling_block:
            reward += self._place_block(instant=True)
        self.last_space_held = space_held
        
        # --- Game Logic Update ---
        self.steps += 1
        self.timer -= 1

        # Update falling block
        if self.falling_block:
            self.falling_block['rect'].y += self.fall_speed
            if self._check_collision():
                reward += self._place_block(instant=False)

        # Update difficulty
        if self.steps > 0 and self.steps % (10 * self.FPS) == 0:
            self.fall_speed += 0.25

        # Update particles
        self._update_particles()
        
        # --- Termination Check ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.stack_height >= self.TARGET_HEIGHT:
                reward += 100
            else:
                reward -= 50
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stack_height": self.stack_height,
            "timer": self.timer / self.FPS,
        }

    # --- Helper Methods: Game Logic ---

    def _spawn_falling_block(self):
        use_small_block = self.stack_height >= 50
        block_width = 60 if use_small_block else 80
        block_height = 15 if use_small_block else 20
        
        start_x = self.np_random.integers(0, self.SCREEN_WIDTH - block_width)
        color = self.BLOCK_COLORS[self.color_index]
        self.color_index = (self.color_index + 1) % len(self.BLOCK_COLORS)

        self.falling_block = {
            'rect': pygame.Rect(start_x, -block_height, block_width, block_height),
            'color': color,
        }
        
        # Spawn obstacles
        if self.stack_height >= 75 and not self.obstacles:
            for _ in range(3):
                obs_y = self.np_random.integers(100, self.GROUND_Y - 100)
                obs_x = self.np_random.integers(0, self.SCREEN_WIDTH - 40)
                self.obstacles.append(pygame.Rect(obs_x, obs_y, 40, 10))

    def _check_collision(self):
        if not self.falling_block:
            return False
        
        block_rect = self.falling_block['rect']
        
        # Ground collision
        if block_rect.bottom >= self.GROUND_Y:
            return True
            
        # Stacked block collision
        for stacked in self.stacked_blocks:
            if block_rect.colliderect(stacked['rect']):
                 if block_rect.bottom > stacked['rect'].top and self.fall_speed > 0:
                    return True

        # Obstacle collision
        for obs in self.obstacles:
            if block_rect.colliderect(obs):
                return True

        return False

    def _place_block(self, instant):
        if not self.falling_block:
            return 0.0

        block = self.falling_block
        landing_y = self.GROUND_Y
        support_block = None

        potential_supports = self.stacked_blocks + [{'rect': obs} for obs in self.obstacles]
        
        # Find the highest surface below the falling block
        highest_support_y = self.GROUND_Y
        for stacked in self.stacked_blocks:
            stacked_rect = stacked['rect']
            # Check if block is horizontally overlapping with a potential support
            if (block['rect'].left < stacked_rect.right and block['rect'].right > stacked_rect.left):
                # Check if the support is below the block and higher than current highest
                if stacked_rect.top >= block['rect'].bottom and stacked_rect.top < highest_support_y:
                    highest_support_y = stacked_rect.top
                    support_block = stacked

        landing_y = highest_support_y
        block['rect'].bottom = landing_y
        
        if instant: # Fast drop
            self._create_particles(block['rect'].midbottom, 10, (200, 200, 255), speed_mult=0.5)

        self.stacked_blocks.append(block)
        self._create_particles(block['rect'].midbottom, 5, block['color'])

        self.stack_height = self._calculate_stack_height()
        
        self.falling_block = None
        
        reward = self._check_chain_reaction(block, support_block)
        
        if not self.game_over:
            self._spawn_falling_block()
        
        return 0.1 + reward

    def _calculate_stack_height(self):
        if not self.stacked_blocks:
            return 0
        min_y = min(b['rect'].top for b in self.stacked_blocks)
        return self.GROUND_Y - min_y

    def _check_chain_reaction(self, placed_block, support_block):
        if not support_block:
            return 0.0

        if abs(placed_block['rect'].centerx - support_block['rect'].centerx) <= self.PERFECT_ALIGN_TOLERANCE:
            chain = [placed_block, support_block]
            current_block = support_block
            
            while True:
                found_next = False
                for b in self.stacked_blocks:
                    if b not in chain and b['rect'].top == current_block['rect'].bottom:
                        if abs(b['rect'].centerx - current_block['rect'].centerx) <= self.PERFECT_ALIGN_TOLERANCE:
                            chain.append(b)
                            current_block = b
                            found_next = True
                            break
                if not found_next:
                    break
            
            if len(chain) >= 2:
                bonus_height = 0
                for block in chain:
                    bonus_height += block['rect'].height
                    self._create_particles(block['rect'].center, 20, (255, 255, 255), speed_mult=2.0)
                
                self.stacked_blocks = [b for b in self.stacked_blocks if b not in chain]
                
                fall_distance = bonus_height
                for b in sorted(self.stacked_blocks, key=lambda x: x['rect'].top):
                    if b['rect'].top < min(c['rect'].top for c in chain):
                        b['rect'].y += fall_distance
                
                self.stack_height = self._calculate_stack_height()
                return 5.0
        return 0.0

    def _check_termination(self):
        return (
            self.timer <= 0 or
            self.stack_height >= self.TARGET_HEIGHT
        )

    # --- Helper Methods: Rendering ---

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game_elements(self):
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))
        pygame.draw.line(self.screen, tuple(min(255, c*1.2) for c in self.COLOR_GROUND), (0, self.GROUND_Y), (self.SCREEN_WIDTH, self.GROUND_Y))

        target_y = self.GROUND_Y - self.TARGET_HEIGHT
        pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (0, target_y), (self.SCREEN_WIDTH, target_y), 2)

        for obs_rect in self.obstacles:
            self._draw_block(self.screen, obs_rect, (40, 50, 60))

        for block in self.stacked_blocks:
            self._draw_block(self.screen, block['rect'], block['color'])
            
        if self.falling_block:
            self._draw_block_with_glow(self.screen, self.falling_block['rect'], self.falling_block['color'])

    def _render_ui(self):
        height_text = self.font_main.render(f"Height: {int(self.stack_height)}/{self.TARGET_HEIGHT}", True, self.COLOR_UI_TEXT)
        self.screen.blit(height_text, (10, 10))
        
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        timer_ratio = max(0, self.timer / (self.MAX_TIME_SECONDS * self.FPS))
        bar_width = self.SCREEN_WIDTH - 20
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR_BG, (10, self.SCREEN_HEIGHT - 15, bar_width, 10), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, (10, self.SCREEN_HEIGHT - 15, bar_width * timer_ratio, 10), border_radius=5)

    def _draw_block(self, surface, rect, color):
        pygame.draw.rect(surface, color, rect)
        highlight = tuple(min(255, c * 1.2) for c in color)
        shadow = tuple(c * 0.7 for c in color)
        pygame.draw.line(surface, highlight, rect.topleft, rect.topright, 2)
        pygame.draw.line(surface, highlight, rect.topleft, rect.bottomleft, 2)
        pygame.draw.line(surface, shadow, rect.bottomleft, rect.bottomright, 2)
        pygame.draw.line(surface, shadow, rect.topright, rect.bottomright, 2)

    def _draw_block_with_glow(self, surface, rect, color):
        glow_radius = 15
        glow_alpha = 90
        glow_surf = pygame.Surface((rect.width + glow_radius * 2, rect.height + glow_radius * 2), pygame.SRCALPHA)
        
        for i in range(glow_radius, 0, -2):
            alpha = glow_alpha * (1 - i / glow_radius)
            pygame.gfxdraw.filled_circle(
                glow_surf,
                rect.width // 2 + glow_radius,
                rect.height // 2 + glow_radius,
                i,
                (*color, alpha)
            )
        
        surface.blit(glow_surf, (rect.x - glow_radius, rect.y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        self._draw_block(surface, rect, color)

    # --- Particle System ---
    def _create_particles(self, pos, count, color, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'max_life': 30,
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
    
    def _render_particles(self):
        for p in self.particles:
            alpha = 255 * (p['life'] / p['max_life'])
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (int(p['size']), int(p['size'])), int(p['size']))
            self.screen.blit(temp_surf, (p['pos'][0] - p['size'], p['pos'][1] - p['size']), special_flags=pygame.BLEND_RGBA_ADD)


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run in the headless environment
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Stacker")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0  # None
        space = 0     # Released
        shift = 0     # Released
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Stack Height: {info['stack_height']}")
            waiting_for_restart = True
            while waiting_for_restart:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_restart = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        waiting_for_restart = False
                clock.tick(env.FPS)
                
        clock.tick(env.FPS)
        
    pygame.quit()