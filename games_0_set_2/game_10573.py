import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:38:14.107417
# Source Brief: brief_00573.md
# Brief Index: 573
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a wobbly block stacking game.

    **Gameplay:**
    The player controls a block at the top of the screen, moving it left and right.
    Pressing 'space' drops the block onto the stack below. The goal is to build
    the tallest, most stable tower possible. Points are awarded based on the
    height of the stack after each successful placement.

    A placement might be unstable if the block is too off-center, causing it to
    wobble. A very poor placement or a bit of bad luck (10% chance on non-perfect
    placements) will cause a collapse. The game ends after 3 collapses or
    when the score reaches 200.

    **Visuals:**
    The game features a clean, minimalist aesthetic with bright, contrasting colors.
    Smooth animations, particle effects for collapses, and a subtle wobble on
    imperfectly placed blocks create a polished and engaging visual experience.

    **Action Space:** `MultiDiscrete([5, 2, 2])`
    - `action[0]`: Movement (0: none, 1: up (no-op), 2: down (no-op), 3: left, 4: right)
    - `action[1]`: Drop Action (0: released, 1: held) - Drop is triggered on press.
    - `action[2]`: Shift Key (0: released, 1: held) - No-op in this game.

    **Observation Space:** `Box(0, 255, (400, 640, 3), uint8)`
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Stack wobbly blocks as high as you can. Move the falling block left and right, "
        "then drop it to build your tower, but be careful of unstable placements!"
    )
    user_guide = "Controls: Use ← and → arrow keys to move the block. Press space to drop it."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.BLOCK_SIZE = 40
        self.GROUND_Y = self.SCREEN_HEIGHT - 40
        self.PLAYER_SPEED = 8
        self.DROP_SPEED = 15
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 200
        self.MAX_COLLAPSES = 3

        # State machine timers
        self.SETTLE_FRAMES = 10
        self.COLLAPSE_FRAMES = 20

        # Physics and Effects
        self.MAX_WOBBLE_AMP = 3.0
        self.WOBBLE_SPEED = 0.2
        self.PARTICLE_COUNT = 40
        self.PARTICLE_LIFESPAN = 45

        # Colors
        self.COLOR_BG = (210, 220, 230)
        self.COLOR_GROUND = (80, 85, 90)
        self.COLOR_UI_TEXT = (50, 55, 60)
        self.BLOCK_COLORS = [
            (231, 76, 60),   # Red
            (46, 204, 113),  # Green
            (52, 152, 219),  # Blue
            (241, 196, 15),   # Yellow
            (155, 89, 182)   # Purple
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # --- State Variables ---
        self.steps = None
        self.score = None
        self.collapses = None
        self.game_over = None
        self.game_state = None
        self.prev_space_held = None
        self.stacked_blocks = None
        self.particles = None
        self.current_block = None
        self.target_y = None
        self.state_timer = None

        self.rng = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.collapses = 0
        self.game_over = False
        self.game_state = "PLACING"
        self.prev_space_held = False
        self.state_timer = 0
        self.particles = []

        # Create the ground plane
        self.stacked_blocks = [{
            'rect': pygame.Rect(0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y),
            'color': self.COLOR_GROUND,
            'wobble_amp': 0, 'wobble_phase': 0, 'is_ground': True
        }]

        self._spawn_new_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        # --- State Machine Logic ---
        if self.game_state == "PLACING":
            if movement == 3: self.current_block['rect'].x -= self.PLAYER_SPEED
            if movement == 4: self.current_block['rect'].x += self.PLAYER_SPEED
            self.current_block['rect'].x = np.clip(self.current_block['rect'].x, 0, self.SCREEN_WIDTH - self.BLOCK_SIZE)

            if space_pressed:
                # sfx: drop_block.play()
                self.game_state = "DROPPING"
                self.target_y = self._get_stack_top_y() - self.BLOCK_SIZE

        elif self.game_state == "DROPPING":
            self.current_block['rect'].y += self.DROP_SPEED
            if self.current_block['rect'].y >= self.target_y:
                self.current_block['rect'].y = self.target_y
                self.game_state = "SETTLING"
                self.state_timer = self.SETTLE_FRAMES

        elif self.game_state == "SETTLING":
            self.state_timer -= 1
            if self.state_timer <= 0:
                is_stable, support_block = self._check_stability()
                if is_stable:
                    # sfx: place_success.play()
                    new_block = {
                        'rect': self.current_block['rect'].copy(),
                        'color': self.current_block['color'],
                        'wobble_amp': self._calculate_wobble(self.current_block, support_block),
                        'wobble_phase': self.rng.uniform(0, 2 * math.pi),
                        'is_ground': False
                    }
                    self.stacked_blocks.append(new_block)
                    
                    placed_height = len(self.stacked_blocks) - 1
                    reward += 0.1 + placed_height
                    self.score += placed_height
                    
                    self._spawn_new_block()
                    self.game_state = "PLACING"
                else:
                    # sfx: collapse.play()
                    self.game_state = "COLLAPSE"
                    self.state_timer = self.COLLAPSE_FRAMES
                    reward -= 5
                    self.collapses += 1
                    self._create_collapse_particles(self.current_block)

        elif self.game_state == "COLLAPSE":
            self.state_timer -= 1
            if self.state_timer <= 0:
                self._spawn_new_block()
                self.game_state = "PLACING"

        self._update_animations()

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100
            if self.collapses >= self.MAX_COLLAPSES:
                reward -= 100
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _check_stability(self):
        placed_rect = self.current_block['rect']
        support_block = self._get_support_block(placed_rect)
        if support_block is None: return False, None # Should not happen

        center_offset = abs(placed_rect.centerx - support_block['rect'].centerx)
        max_allowed_offset = (support_block['rect'].width / 2) if not support_block['is_ground'] else self.SCREEN_WIDTH

        if center_offset > max_allowed_offset:
            return False, support_block

        is_perfect = center_offset < 2
        if not is_perfect and self.rng.random() < 0.10:
            return False, support_block

        return True, support_block

    def _get_support_block(self, placed_rect):
        highest_support = None
        highest_y = -1
        for block in self.stacked_blocks:
            # Check for horizontal overlap and if the block is below the placed one
            if (placed_rect.left < block['rect'].right and placed_rect.right > block['rect'].left and
                    placed_rect.bottom >= block['rect'].top):
                if block['rect'].top > highest_y:
                    highest_y = block['rect'].top
                    highest_support = block
        return highest_support

    def _calculate_wobble(self, placed_block, support_block):
        if support_block is None or support_block['is_ground']: return 0
        center_offset = abs(placed_block['rect'].centerx - support_block['rect'].centerx)
        max_offset = support_block['rect'].width / 2
        wobble = (center_offset / max(1, max_offset)) * self.MAX_WOBBLE_AMP
        return wobble

    def _get_stack_top_y(self):
        if len(self.stacked_blocks) == 1: return self.GROUND_Y
        highest_y = min(b['rect'].top for b in self.stacked_blocks if not b['is_ground'])
        return highest_y

    def _spawn_new_block(self):
        self.current_block = {
            'rect': pygame.Rect(self.SCREEN_WIDTH / 2 - self.BLOCK_SIZE / 2, 50, self.BLOCK_SIZE, self.BLOCK_SIZE),
            'color': self.BLOCK_COLORS[self.rng.integers(0, len(self.BLOCK_COLORS))],
        }
    
    def _create_collapse_particles(self, block):
        cx, cy = block['rect'].center
        for _ in range(self.PARTICLE_COUNT):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 5)
            self.particles.append({
                'pos': [cx, cy],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed - 2],
                'lifespan': self.PARTICLE_LIFESPAN,
                'color': block['color'],
                'size': self.rng.integers(3, 7)
            })

    def _update_animations(self):
        # Update block wobbles
        for block in self.stacked_blocks:
            if not block['is_ground']:
                block['wobble_phase'] += self.WOBBLE_SPEED

        # Update particles
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['lifespan'] -= 1
            if p['lifespan'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _check_termination(self):
        return (self.score >= self.WIN_SCORE or
                self.collapses >= self.MAX_COLLAPSES)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw stacked blocks
        for block in self.stacked_blocks:
            if block['is_ground']:
                pygame.draw.rect(self.screen, block['color'], block['rect'])
            else:
                wobble_offset = math.sin(block['wobble_phase']) * block['wobble_amp']
                draw_rect = block['rect'].move(wobble_offset, 0)
                pygame.draw.rect(self.screen, block['color'], draw_rect)
                pygame.draw.rect(self.screen, tuple(max(0, c-40) for c in block['color']), draw_rect, 2)

        # Draw current block with glow
        if self.game_state in ["PLACING", "DROPPING"]:
            r = self.current_block['rect']
            glow_color = self.current_block['color'] + (70,) # Add alpha
            for i in range(10, 0, -2):
                glow_surf = pygame.Surface((r.width + i, r.height + i), pygame.SRCALPHA)
                pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), border_radius=5)
                self.screen.blit(glow_surf, (r.x - i/2, r.y - i/2))
            
            pygame.draw.rect(self.screen, self.current_block['color'], r)
            pygame.draw.rect(self.screen, tuple(max(0, c-40) for c in self.current_block['color']), r, 2)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / self.PARTICLE_LIFESPAN))
            color = p['color'] + (alpha,)
            try:
                pygame.gfxdraw.box(self.screen, (int(p['pos'][0]), int(p['pos'][1]), p['size'], p['size']), color)
            except TypeError: # Sometimes color can have invalid alpha
                pass


    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Collapses display
        collapse_text = self.font_large.render(f"COLLAPSES: {self.collapses}/{self.MAX_COLLAPSES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(collapse_text, (self.SCREEN_WIDTH - collapse_text.get_width() - 15, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            result_text = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            text_surf = self.font_large.render(result_text, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)


    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "collapses": self.collapses}

    def close(self):
        pygame.quit()

# To run and play the game manually
if __name__ == "__main__":
    # Un-comment the line below to run with a display
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset(seed=42)
    done = False
    
    # Pygame setup for manual play
    pygame.display.set_caption("Wobbly Stacker")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action.fill(0) # Start with no action

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Map keyboard keys to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # No movement
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action[0] = 4
        
        # Space button
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        
        # Shift button (unused but part of the space)
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Run at 30 FPS

    env.close()