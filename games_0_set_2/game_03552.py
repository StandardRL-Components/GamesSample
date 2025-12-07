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

    # Short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the falling block. Press Space to drop it quickly."
    )

    # Short, user-facing description of the game:
    game_description = (
        "Build the tallest tower you can in 60 seconds by strategically stacking falling blocks. "
        "Special blocks give bonuses but might affect stability!"
    )

    # Frames auto-advance for this real-time game.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and Game Dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GROUND_Y = self.HEIGHT - 40
        self.FPS = 30

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (35, 40, 60)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_TIMER_WARN = (255, 100, 100)
        self.COLOR_BLOCK_RED = (230, 50, 50)
        self.COLOR_BLOCK_GREEN = (50, 230, 50)
        self.COLOR_BLOCK_BLUE = (50, 120, 230)
        self.COLOR_BLOCK_BASE = (100, 100, 120)
        self.COLOR_GHOST = (255, 255, 255, 60)

        # Block Properties
        self.BLOCK_HEIGHT = 20
        self.BLOCK_WIDTH_NORMAL = 60
        self.BLOCK_WIDTH_WIDE = 100
        self.BLOCK_MOVE_SPEED = 8
        self.INITIAL_FALL_SPEED = 1.5
        self.FALL_SPEED_INCREMENT = 0.5

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_height = pygame.font.SysFont("Consolas", 24, bold=True)

        # Initialize state variables
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        self.stacked_blocks = []
        self.falling_block = None
        self.particles = []
        self.tower_height = 0
        self.fall_speed = 0
        self.win_condition_met = False

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None or seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        self.timer = 60 * self.FPS
        self.tower_height = 1
        self.fall_speed = self.INITIAL_FALL_SPEED

        # Create the base block
        base_width = 200
        base_rect = pygame.Rect(
            (self.WIDTH - base_width) / 2,
            self.GROUND_Y - self.BLOCK_HEIGHT,
            base_width,
            self.BLOCK_HEIGHT
        )
        self.stacked_blocks = [{
            'rect': base_rect,
            'type': 'base',
            'color': self.COLOR_BLOCK_BASE,
            'wobble_offset': 0
        }]

        self.particles = []
        self._spawn_new_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1

        # --- Action Handling ---
        if self.falling_block:
            if movement == 3:  # Left
                self.falling_block['rect'].x -= self.BLOCK_MOVE_SPEED
            elif movement == 4:  # Right
                self.falling_block['rect'].x += self.BLOCK_MOVE_SPEED

            # Clamp to screen
            self.falling_block['rect'].left = max(0, self.falling_block['rect'].left)
            self.falling_block['rect'].right = min(self.WIDTH, self.falling_block['rect'].right)

            # Fast drop
            if space_held:
                top_block = self.stacked_blocks[-1]
                self.falling_block['rect'].bottom = top_block['rect'].top
                # Sound: Fast drop whoosh

        # --- Game Logic Update ---
        self.steps += 1
        self.timer -= 1

        # Update falling block
        if self.falling_block:
            self.falling_block['rect'].y += self.fall_speed

            # Check for landing
            top_block = self.stacked_blocks[-1]
            if self.falling_block['rect'].colliderect(top_block['rect']) and self.falling_block['rect'].bottom > top_block['rect'].top:
                self.falling_block['rect'].bottom = top_block['rect'].top
                reward += self._place_block()

        # Update particles
        self._update_particles()

        # Update tower physics and check for lean
        lean_penalty = self._update_tower_physics()
        reward -= lean_penalty

        # Check termination conditions
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.win_condition_met:
                reward += 100  # Win reward
                # Sound: Victory fanfare
            elif self.timer <= 0:
                pass  # No penalty for timeout
            else:  # Collapse
                reward -= 50  # Collapse penalty
                # Sound: Tower crumbling

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _place_block(self):
        placed_block_info = self.falling_block
        self.falling_block = None

        # Sound: Block place 'thud'

        # Calculate placement properties
        support_block = self.stacked_blocks[-1]
        offset = placed_block_info['rect'].centerx - support_block['rect'].centerx
        placed_block_info['wobble_offset'] = offset

        self.stacked_blocks.append(placed_block_info)

        # Calculate rewards and score
        placement_reward = 0.1
        self.score += 10  # Base score for placement

        if placed_block_info['type'] == 'green':
            placement_reward += 1
            self.score += 20
            # Sound: Positive chime for green block
        elif placed_block_info['type'] == 'blue':
            placement_reward += 2
            self.tower_height += 4  # +1 is implicit, so +4 here
            self.score += 50
            # Sound: Power-up sound for blue block

        self.tower_height += 1

        # Increase difficulty
        if self.tower_height % 10 == 0:
            self.fall_speed += self.FALL_SPEED_INCREMENT

        # Spawn particles
        self._create_particles(placed_block_info['rect'].midbottom, placed_block_info['color'])

        # Spawn next block if game is not over
        if self.tower_height < 20:
            self._spawn_new_block()
        else:
            self.win_condition_met = True

        return placement_reward

    def _spawn_new_block(self):
        block_type_roll = self.np_random.random()
        if block_type_roll < 0.15:  # 15% chance for green
            block_type = 'green'
            width = self.BLOCK_WIDTH_WIDE
            color = self.COLOR_BLOCK_GREEN
        elif block_type_roll < 0.30:  # 15% chance for blue
            block_type = 'blue'
            width = self.BLOCK_WIDTH_NORMAL
            color = self.COLOR_BLOCK_BLUE
        else:  # 70% chance for red
            block_type = 'red'
            width = self.BLOCK_WIDTH_NORMAL
            color = self.COLOR_BLOCK_RED

        start_x = self.np_random.integers(self.WIDTH // 4, self.WIDTH * 3 // 4)
        rect = pygame.Rect(start_x, -self.BLOCK_HEIGHT, width, self.BLOCK_HEIGHT)

        self.falling_block = {
            'rect': rect,
            'type': block_type,
            'color': color,
        }

    def _update_tower_physics(self):
        # --- Check for collapse ---
        # A block collapses if its center of mass is outside the support of the block below it.
        # We check this from the top down.
        for i in range(len(self.stacked_blocks) - 1, 0, -1):
            block = self.stacked_blocks[i]
            support = self.stacked_blocks[i - 1]

            # Check if block has fallen off the support
            if abs(block['rect'].centerx - support['rect'].centerx) > support['rect'].width / 2:
                self.game_over = True
                self._create_collapse_particles(block['rect'], block['color'])
                self.stacked_blocks = self.stacked_blocks[:i]  # The rest of the tower collapses
                return 0  # No lean penalty on collapse frame

        # --- Calculate lean penalty ---
        # Calculate total tower center of mass relative to base
        total_mass_moment = 0
        total_mass = 0
        base_center_x = self.stacked_blocks[0]['rect'].centerx

        for block in self.stacked_blocks[1:]:  # Exclude base
            mass = block['rect'].width
            # Blue blocks are "unstable", effectively heavier for CoM calculation
            if block['type'] == 'blue':
                mass *= 1.5
            total_mass_moment += (block['rect'].centerx - base_center_x) * mass
            total_mass += mass

        if total_mass > 0:
            com_offset = total_mass_moment / total_mass
            base_width = self.stacked_blocks[0]['rect'].width
            # Penalty is proportional to how far the CoM has shifted, as a percentage of support width
            if abs(com_offset) > base_width * 0.1:  # Only penalize significant lean
                return 0.02
        return 0

    def _check_termination(self):
        if self.game_over:
            return True
        if self.win_condition_met:
            return True
        if self.timer <= 0:
            return True
        if self.steps >= 1800:  # 60s * 30fps
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": self.tower_height,
            "time_left": self.timer / self.FPS
        }

    def _render_background(self):
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
        pygame.draw.line(self.screen, self.COLOR_BLOCK_BASE, (0, self.GROUND_Y), (self.WIDTH, self.GROUND_Y), 3)

    def _render_game(self):
        # Render stacked blocks with a visual wobble
        wobble_angle = math.sin(self.steps * 0.1) * 0.5  # A gentle global sway

        for i, block_info in enumerate(self.stacked_blocks):
            rect = block_info['rect']

            # Simple wobble effect based on height
            height_wobble = (i / len(self.stacked_blocks)) * wobble_angle if len(self.stacked_blocks) > 1 else 0

            # Create a rotated surface for wobble
            surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(surf, block_info['color'], (0, 0, rect.width, rect.height))
            pygame.draw.rect(surf, tuple(max(0, c - 40) for c in block_info['color']),
                             (0, 0, rect.width, rect.height), 2)

            rotated_surf = pygame.transform.rotate(surf, height_wobble)
            new_rect = rotated_surf.get_rect(center=rect.center)

            self.screen.blit(rotated_surf, new_rect.topleft)

        # Render falling block and its ghost
        if self.falling_block:
            # Ghost block
            top_block = self.stacked_blocks[-1]
            ghost_rect = self.falling_block['rect'].copy()
            ghost_rect.bottom = top_block['rect'].top

            s = pygame.Surface(ghost_rect.size, pygame.SRCALPHA)
            s.fill(self.COLOR_GHOST)
            self.screen.blit(s, ghost_rect.topleft)

            # Actual block
            fb_info = self.falling_block
            pygame.draw.rect(self.screen, fb_info['color'], fb_info['rect'])
            pygame.draw.rect(self.screen, tuple(max(0, c - 40) for c in fb_info['color']), fb_info['rect'], 2)

        # Render particles
        self._render_particles()

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, self.timer / self.FPS)
        timer_color = self.COLOR_UI_TEXT if time_left > 10 else self.COLOR_TIMER_WARN
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Height
        height_str = f"HEIGHT: {self.tower_height} / 20"
        height_text = self.font_height.render(height_str, True, self.COLOR_UI_TEXT)
        text_rect = height_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 20))
        self.screen.blit(height_text, text_rect)

    def _create_particles(self, pos, color, count=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 25)
            size = self.np_random.integers(2, 5)
            self.particles.append(
                {'pos': list(pos), 'vel': vel, 'life': life, 'max_life': life, 'size': size, 'color': color})

    def _create_collapse_particles(self, rect, color):
        for _ in range(50):
            pos = [self.np_random.uniform(rect.left, rect.right), self.np_random.uniform(rect.top, rect.bottom)]
            angle = self.np_random.uniform(math.pi, 2 * math.pi)  # Downward explosion
            speed = self.np_random.uniform(2, 6)
            vel = [math.cos(angle) * speed, abs(math.sin(angle)) * speed + 1]  # Add gravity
            life = self.np_random.integers(30, 60)
            size = self.np_random.integers(3, 7)
            self.particles.append(
                {'pos': pos, 'vel': vel, 'life': life, 'max_life': life, 'size': size, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)

            s = pygame.Surface((p['size'] * 2, p['size'] * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
            self.screen.blit(s, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))


if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Setup Pygame window for human play
    pygame.display.init()
    pygame.font.init()
    pygame.display.set_caption("Tower Stacker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        # Action defaults
        movement = 0  # none
        space = 0  # released
        shift = 0  # released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        if keys[pygame.K_SPACE]:
            space = 1

        action = [movement, space, shift]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Height: {info['height']}")
            print("Press 'R' to restart.")

        clock.tick(env.FPS)

    pygame.quit()