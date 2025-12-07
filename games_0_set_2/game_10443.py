import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:29:16.693019
# Source Brief: brief_00443.md
# Brief Index: 443
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Guide falling blocks onto a target platform. Land blocks quickly to build a score multiplier, but be careful not to miss!"
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to move the falling block. Press ↓ to make it fall faster."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.PLAY_AREA_WIDTH, self.PLAY_AREA_HEIGHT = 250, 350
        self.PLAY_AREA_X_START = (self.WIDTH - self.PLAY_AREA_WIDTH) // 2
        self.PLAY_AREA_Y_START = (self.HEIGHT - self.PLAY_AREA_HEIGHT) // 2
        self.PLAY_AREA_X_END = self.PLAY_AREA_X_START + self.PLAY_AREA_WIDTH
        self.PLAY_AREA_Y_END = self.PLAY_AREA_Y_START + self.PLAY_AREA_HEIGHT

        self.PLAYER_SPEED = 4.0
        self.DOWN_ACCEL = 3.0
        self.BLOCK_SIZE = 20
        self.TARGET_HEIGHT = 10
        self.TARGET_WIDTH = 80
        
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 1500
        self.CHAIN_WINDOW_STEPS = 45 # 1.5 seconds at 30 FPS

        # --- Visuals ---
        self.COLOR_BG = (26, 26, 46) # #1a1a2e
        self.COLOR_BORDER = (240, 240, 240)
        self.COLOR_TARGET = (0, 255, 127) # #00ff7f
        self.COLOR_TARGET_FLASH = (255, 255, 255)
        self.BLOCK_COLORS = [
            (233, 69, 96),   # #e94560
            (58, 134, 255),  # #3a86ff
            (255, 204, 0)    # #ffcc00
        ]
        self.COLOR_UI_TEXT = (240, 240, 240)

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        # --- Game State Variables ---
        # These are initialized here to avoid AttributeError, but properly set in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.chain_multiplier = 1
        self.last_hit_time = 0
        self.base_block_speed = 0.0
        self.difficulty_level = 0
        self.block = {}
        self.particles = []
        self.target_flash_frames = 0
        self.last_block_x_for_reward = 0.0
        
        self.target = pygame.Rect(
            self.PLAY_AREA_X_START + (self.PLAY_AREA_WIDTH - self.TARGET_WIDTH) // 2,
            self.PLAY_AREA_Y_END - self.TARGET_HEIGHT,
            self.TARGET_WIDTH,
            self.TARGET_HEIGHT
        )
        
        # Initialize state
        # self.reset() is called by the environment wrapper
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.chain_multiplier = 1
        self.last_hit_time = -self.CHAIN_WINDOW_STEPS * 2 # Ensure first hit doesn't chain
        self.base_block_speed = 1.0
        self.difficulty_level = 0
        self.particles = []
        self.target_flash_frames = 0

        self._reset_block()
        
        return self._get_observation(), self._get_info()

    def _reset_block(self):
        """Spawns a new block at a random horizontal position at the top."""
        x_pos = self.np_random.uniform(
            self.PLAY_AREA_X_START, 
            self.PLAY_AREA_X_END - self.BLOCK_SIZE
        )
        self.block = {
            'x': x_pos,
            'y': float(self.PLAY_AREA_Y_START),
            'color': random.choice(self.BLOCK_COLORS),
            'rect': pygame.Rect(int(x_pos), self.PLAY_AREA_Y_START, self.BLOCK_SIZE, self.BLOCK_SIZE)
        }
        self.last_block_x_for_reward = self.block['x']
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1
        # shift_held = action[2] == 1

        # --- Update Game Logic ---
        self.steps += 1
        
        # 1. Handle player input and block movement
        self._update_block(movement)

        # 2. Calculate reward for this step
        reward = self._calculate_reward()

        # 3. Check for landing and scoring
        landed, hit_target = self._check_landing()
        if landed:
            if hit_target:
                # SFX: Positive hit sound
                hit_score = self._process_hit()
                reward += hit_score # Use scaled score as reward
            else:
                # SFX: Miss sound
                self._process_miss()
            self._reset_block()

        # 4. Update visual effects
        self._update_effects()
        
        # 5. Check for termination
        terminated = self._check_termination()
        if terminated and self.score >= self.WIN_SCORE:
            reward += 100 # Goal-oriented reward

        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_block(self, movement):
        """Updates the block's position based on player action and gravity."""
        # Horizontal movement
        dx = 0
        if movement == 3: # left
            dx = -self.PLAYER_SPEED
        elif movement == 4: # right
            dx = self.PLAYER_SPEED
        
        self.block['x'] += dx
        self.block['x'] = np.clip(
            self.block['x'], 
            self.PLAY_AREA_X_START, 
            self.PLAY_AREA_X_END - self.BLOCK_SIZE
        )

        # Vertical movement
        effective_speed_y = self.base_block_speed
        if movement == 2: # down
            effective_speed_y += self.DOWN_ACCEL
        
        self.block['y'] += effective_speed_y
        
        # Update rect for collision and rendering
        self.block['rect'].x = int(self.block['x'])
        self.block['rect'].y = int(self.block['y'])

    def _check_landing(self):
        """Checks if the block has reached the bottom of the play area."""
        if self.block['rect'].bottom >= self.PLAY_AREA_Y_END:
            hit_target = self.block['rect'].colliderect(self.target)
            return True, hit_target
        return False, False

    def _process_hit(self):
        """Handles the logic for a successful hit on the target."""
        if self.steps - self.last_hit_time < self.CHAIN_WINDOW_STEPS:
            self.chain_multiplier += 1
            # SFX: Chain increase sound
        else:
            self.chain_multiplier = 1
        
        self.last_hit_time = self.steps
        
        hit_score = 20 * self.chain_multiplier
        self.score += hit_score
        
        self.target_flash_frames = 15
        self._create_particles(self.block['rect'].midbottom, self.block['color'])
        
        # Difficulty scaling
        new_difficulty_level = self.score // 500
        if new_difficulty_level > self.difficulty_level:
            self.difficulty_level = new_difficulty_level
            self.base_block_speed += 0.2
            # SFX: Level up/speed up sound
        
        return 20.0 # Event-based reward

    def _process_miss(self):
        """Resets the chain multiplier on a miss."""
        self.chain_multiplier = 1

    def _update_effects(self):
        """Updates particles and other visual effects."""
        # Update particles
        live_particles = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1  # Gravity on particles
            p['life'] -= 1
            if p['life'] > 0:
                live_particles.append(p)
        self.particles = live_particles

        # Update target flash
        if self.target_flash_frames > 0:
            self.target_flash_frames -= 1

    def _calculate_reward(self):
        """Calculates the continuous reward for moving towards the target."""
        # Reward for moving closer to the target's center
        current_dist = abs(self.block['rect'].centerx - self.target.centerx)
        last_dist = abs(self.last_block_x_for_reward + self.BLOCK_SIZE/2 - self.target.centerx)
        
        reward = (last_dist - current_dist) * 0.1 # Small incentive
        
        self.last_block_x_for_reward = self.block['x']
        return reward

    def _check_termination(self):
        """Checks if the episode should end."""
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        """Renders the game state to a numpy array."""
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders all non-UI game elements."""
        # Draw play area border
        pygame.draw.rect(self.screen, self.COLOR_BORDER, 
                         (self.PLAY_AREA_X_START-2, self.PLAY_AREA_Y_START-2, 
                          self.PLAY_AREA_WIDTH+4, self.PLAY_AREA_HEIGHT+4), 2, border_radius=5)

        # Draw target area
        target_color = self.COLOR_TARGET_FLASH if self.target_flash_frames > 5 else self.COLOR_TARGET
        pygame.draw.rect(self.screen, target_color, self.target, border_radius=3)
        if self.target_flash_frames > 0:
            # Glow effect
            glow_radius = self.target_flash_frames
            glow_alpha = int(128 * (self.target_flash_frames / 15))
            glow_surf = pygame.Surface((self.TARGET_WIDTH + glow_radius*2, self.TARGET_HEIGHT + glow_radius*2), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*self.COLOR_TARGET_FLASH, glow_alpha), glow_surf.get_rect(), border_radius=glow_radius)
            self.screen.blit(glow_surf, (self.target.x - glow_radius, self.target.y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)


        # Draw falling block
        if 'rect' in self.block: # Ensure block has been initialized
            block_rect = self.block['rect']
            pygame.draw.rect(self.screen, self.block['color'], block_rect, border_radius=3)
            pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in self.block['color']), block_rect, 2, border_radius=3)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['x'] - p['size']), int(p['y'] - p['size'])), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        """Renders the UI elements like score and multiplier."""
        # Score display
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Chain multiplier display
        if self.chain_multiplier > 1:
            chain_text = self.font_small.render(f"x{self.chain_multiplier} CHAIN!", True, self.COLOR_TARGET)
            self.screen.blit(chain_text, (20, 50))
        
        # Steps display
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 20, 20))

    def _create_particles(self, pos, color):
        """Generates a burst of particles."""
        for _ in range(20):
            angle = self.np_random.uniform(math.pi, 2 * math.pi) # Upward burst
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(20, 40)
            self.particles.append({
                'x': pos[0],
                'y': pos[1],
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "chain_multiplier": self.chain_multiplier,
            "difficulty_level": self.difficulty_level,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # Ensure the display is not dummy for manual play
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    # Pygame setup for manual play
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Drop")
    clock = pygame.time.Clock()

    total_reward = 0
    
    while not terminated and not truncated:
        # --- Action Mapping for Manual Play ---
        movement = 0 # none
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over!")
    print(f"Final Score: {info['score']}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Steps: {info['steps']}")

    env.close()