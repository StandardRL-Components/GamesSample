import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:25:09.296159
# Source Brief: brief_00998.md
# Brief Index: 998
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent must stack four falling blocks perfectly.
    The agent controls one block's magnetic property and can move it horizontally
    to attract and align the other blocks for a perfect vertical stack.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Stack four falling blocks perfectly by controlling a magnetic block. "
        "Attract and align the other blocks to create a stable tower before time runs out."
    )
    user_guide = (
        "Use ←→ arrow keys to move the magnetic block horizontally. "
        "Press space to toggle the magnet on and off to attract other blocks."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 900  # 30 seconds at 30 FPS

    # Colors
    COLOR_BG = (26, 26, 26)  # Dark Gray
    COLOR_UI = (220, 220, 220)
    COLOR_TARGET_ZONE = (70, 70, 70)
    BLOCK_COLORS = [
        (255, 70, 70),   # Red
        (70, 255, 70),   # Green
        (70, 130, 255),  # Blue
        (255, 220, 70),  # Yellow
    ]
    
    # Game Parameters
    BLOCK_SIZE = 32
    STACK_TOLERANCE = 12
    HORIZONTAL_ACCEL = 0.5
    FRICTION = 0.90
    MAGNETIC_STRENGTH = 2000.0
    PARTICLE_LIFESPAN = 20

    class Particle:
        """A simple particle for visual effects."""
        def __init__(self, pos, color):
            self.pos = list(pos)
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.lifespan = GameEnv.PARTICLE_LIFESPAN
            self.color = color

    class Block:
        """Represents a single block in the game."""
        def __init__(self, x, y, color, base_speed, is_magnet_target=False):
            self.pos = np.array([x, y], dtype=float)
            self.vel = np.array([0.0, base_speed], dtype=float)
            self.color = color
            self.is_magnet_target = is_magnet_target
            self.is_magnetized = False
            self.is_stacked = False
            self.size = GameEnv.BLOCK_SIZE

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Consolas', 24, bold=True)
        self.game_over_font = pygame.font.SysFont('Consolas', 48, bold=True)
        
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.was_space_held = False
        
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.was_space_held = False
        self.particles.clear()
        
        self.blocks = []
        base_speeds = [1.0, 1.5, 2.0, 2.5]
        random.shuffle(base_speeds)
        
        for i in range(4):
            x_pos = self.np_random.uniform(self.BLOCK_SIZE, self.WIDTH - self.BLOCK_SIZE)
            y_pos = self.np_random.uniform(20, 100)
            is_magnet_target = (base_speeds[i] == 2.5) # Fastest block is the target
            self.blocks.append(self.Block(
                x=x_pos, y=y_pos, color=self.BLOCK_COLORS[i],
                base_speed=base_speeds[i], is_magnet_target=is_magnet_target
            ))

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        
        self._handle_input(action)
        self._update_physics()
        reward, terminated = self._calculate_reward_and_check_termination()
        
        self.score += reward
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Toggle magnetization on space press (0 -> 1 transition)
        is_space_pressed = space_held and not self.was_space_held
        if is_space_pressed:
            # SFX: Magnet activate/deactivate
            for block in self.blocks:
                if block.is_magnet_target:
                    block.is_magnetized = not block.is_magnetized
        self.was_space_held = bool(space_held)

        # Apply movement to the magnetized block
        magnet_block = next((b for b in self.blocks if b.is_magnetized), None)
        if magnet_block and not magnet_block.is_stacked:
            if movement == 3:  # Left
                magnet_block.vel[0] -= self.HORIZONTAL_ACCEL
            elif movement == 4:  # Right
                magnet_block.vel[0] += self.HORIZONTAL_ACCEL

    def _update_physics(self):
        magnet_block = next((b for b in self.blocks if b.is_magnetized), None)

        for block in self.blocks:
            if block.is_stacked:
                block.vel = np.array([0.0, 0.0])
                continue

            # Apply magnetic force if a magnet is active
            if magnet_block and block is not magnet_block:
                direction_vec = magnet_block.pos - block.pos
                distance_sq = np.sum(direction_vec**2)
                if distance_sq > 1: # Avoid division by zero
                    force_magnitude = self.MAGNETIC_STRENGTH / distance_sq
                    force_vec = (direction_vec / np.sqrt(distance_sq)) * force_magnitude
                    block.vel += force_vec / self.FPS

            # Apply friction
            block.vel[0] *= self.FRICTION
            
            # Update position
            block.pos += block.vel

            # Horizontal wrapping
            if block.pos[0] < 0: block.pos[0] += self.WIDTH
            if block.pos[0] > self.WIDTH: block.pos[0] -= self.WIDTH

        self._handle_stacking()
        self._update_particles()

    def _handle_stacking(self):
        stacked_blocks = [b for b in self.blocks if b.is_stacked]
        stack_base_y = self.HEIGHT - self.BLOCK_SIZE // 2
        
        if stacked_blocks:
            highest_stacked_y = min(b.pos[1] for b in stacked_blocks)
            stack_target_y = highest_stacked_y - self.BLOCK_SIZE
        else:
            stack_target_y = stack_base_y
        
        for block in self.blocks:
            if not block.is_stacked and block.pos[1] >= stack_target_y:
                # SFX: Block stack
                block.is_stacked = True
                block.pos[1] = stack_target_y
                block.vel = np.array([0.0, 0.0])
                for _ in range(30):
                    self.particles.append(self.Particle(block.pos, block.color))
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.pos[0] += p.vel[0]
            p.pos[1] += p.vel[1]
            p.lifespan -= 1
            
    def _calculate_reward_and_check_termination(self):
        reward = 0.0
        terminated = False

        # Continuous alignment reward
        unstacked_blocks = [b for b in self.blocks if not b.is_stacked]
        if len(unstacked_blocks) > 1:
            x_positions = [b.pos[0] for b in unstacked_blocks]
            alignment_error = np.std(x_positions)
            reward += max(0, 1.0 - alignment_error / (self.WIDTH / 4)) * 0.01

        # Stacking reward
        num_stacked = sum(1 for b in self.blocks if b.is_stacked)
        if num_stacked == 4:
            x_coords = [b.pos[0] for b in self.blocks]
            if max(x_coords) - min(x_coords) < self.STACK_TOLERANCE:
                # VICTORY
                reward += 100
                terminated = True
            else:
                # FAILED STACK
                reward -= 10
                terminated = True
        
        # Timeout
        if self.steps >= self.MAX_STEPS:
            reward -= 10
            terminated = True
            
        return reward, terminated

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw target zone
        target_width = self.STACK_TOLERANCE * 2
        target_x = self.WIDTH / 2 - target_width / 2
        pygame.draw.rect(self.screen, self.COLOR_TARGET_ZONE, 
                         (target_x, self.HEIGHT - self.BLOCK_SIZE * 4, target_width, self.BLOCK_SIZE * 4))

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p.lifespan / self.PARTICLE_LIFESPAN))
            color = (*p.color, alpha)
            temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, 3, 3))
            self.screen.blit(temp_surf, (int(p.pos[0]), int(p.pos[1])))

        # Draw blocks and effects
        for block in self.blocks:
            # Magnetism Aura
            if block.is_magnetized:
                glow_surf = pygame.Surface((self.BLOCK_SIZE * 4, self.BLOCK_SIZE * 4), pygame.SRCALPHA)
                center = (self.BLOCK_SIZE * 2, self.BLOCK_SIZE * 2)
                for i in range(15, 0, -1):
                    alpha = 100 - (i / 15 * 100)
                    radius = int(self.BLOCK_SIZE * 0.5 + i * 2.5)
                    pygame.gfxdraw.aacircle(glow_surf, center[0], center[1], radius, (255, 255, 255, alpha))
                self.screen.blit(glow_surf, (int(block.pos[0] - center[0]), int(block.pos[1] - center[1])), special_flags=pygame.BLEND_RGBA_ADD)

            # Block body
            rect = pygame.Rect(0, 0, block.size, block.size)
            rect.center = (int(block.pos[0]), int(block.pos[1]))
            
            # Bevel effect
            darker_color = tuple(max(0, c - 50) for c in block.color)
            pygame.draw.rect(self.screen, darker_color, rect, border_radius=4)
            inner_rect = rect.inflate(-6, -6)
            pygame.draw.rect(self.screen, block.color, inner_rect, border_radius=3)

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_color = self.COLOR_UI if time_left > 5 else (255, 100, 100)
        timer_text = self.font.render(f"TIME: {max(0, time_left):.1f}", True, time_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))
        
        if self.game_over:
            num_stacked = sum(1 for b in self.blocks if b.is_stacked)
            if num_stacked == 4:
                x_coords = [b.pos[0] for b in self.blocks]
                if max(x_coords) - min(x_coords) < self.STACK_TOLERANCE:
                    msg = "PERFECT STACK!"
                    color = (100, 255, 100)
                else:
                    msg = "IMPERFECT STACK"
                    color = (255, 180, 0)
            else:
                msg = "TIME UP"
                color = (255, 100, 100)
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.game_over_font.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # This block is for manual play and visualization, and is not part of the
    # required environment implementation. It's safe to run.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use a visible driver for manual play
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    obs, info = env.reset()
    done = False
    
    # Pygame setup for rendering
    pygame.init()
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Stack Attack")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Action defaults
        movement = 0 # none
        space = 0    # released
        shift = 0    # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            
        clock.tick(GameEnv.FPS)
        
    env.close()