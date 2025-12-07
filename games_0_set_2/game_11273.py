import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:46:13.805149
# Source Brief: brief_01273.md
# Brief Index: 1273
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from enum import Enum

class BlockState(Enum):
    HELD = 1
    FALLING = 2
    PLACED = 3

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Drop blocks from a moving robotic arm onto a conveyor belt below. "
        "Time your drops perfectly to land blocks in the target zone and increase your score."
    )
    user_guide = "Controls: Use ←→ arrow keys to move the arm. Press space to drop the block."
    auto_advance = True

    # --- CONSTANTS ---
    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_ARM = (50, 150, 255)
    COLOR_ARM_GLOW = (50, 150, 255, 50)
    COLOR_BLOCK = (255, 50, 50)
    COLOR_BLOCK_GLOW = (255, 50, 50, 60)
    COLOR_CONVEYOR = (200, 180, 0)
    COLOR_CONVEYOR_TREADS = (160, 140, 0)
    COLOR_SUCCESS = (50, 255, 50)
    COLOR_FAILURE = (255, 50, 50)
    COLOR_SPARK = (255, 200, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TARGET_ZONE = (0, 255, 0, 90) # Semi-transparent green

    # Dimensions & Speeds
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    ARM_WIDTH = 80
    ARM_HEIGHT = 20
    ARM_SPEED = 5.0
    BLOCK_SIZE = 30
    BLOCK_FALL_SPEED = 4.0
    CONVEYOR_Y = 350
    CONVEYOR_HEIGHT = 50
    CONVEYOR_TREAD_SPACING = 40
    BASE_CONVEYOR_SPEED = 2.0
    TARGET_ZONE_WIDTH = 40

    # Game Rules
    WIN_SCORE = 20
    MAX_STEPS = 1500 # Increased for longer play potential

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)

        self.arm_pos = pygame.Vector2(0, 0)
        self.block_pos = pygame.Vector2(0, 0)
        self.block_state = BlockState.HELD
        self.conveyor_speed = 0.0
        self.conveyor_tread_offset = 0.0
        self.placed_blocks = []
        self.particles = []
        self.feedback_flash = {"alpha": 0, "color": (0,0,0)}
        self.prev_space_held = False
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False

        # self.reset() is called by the wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False

        self.arm_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, 50)
        self.conveyor_speed = self.BASE_CONVEYOR_SPEED
        self.conveyor_tread_offset = 0.0
        
        self.placed_blocks = []
        self.particles = []
        self.feedback_flash = {"alpha": 0, "color": (0,0,0)}
        self.prev_space_held = False

        self._spawn_new_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        self.game_over = False

        # --- Update Game Logic ---
        self._update_arm(movement)
        self._handle_input(space_held)
        self._update_block()
        self._update_conveyor()
        self._update_particles()
        self._update_feedback_flash()

        # --- Collision & Scoring ---
        event_reward = self._check_block_landing()
        reward += event_reward

        # --- Continuous Reward ---
        if self.block_state == BlockState.FALLING:
            target_x = self.SCREEN_WIDTH / 2 # Target is always center screen when dropped
            if abs(self.arm_pos.x - target_x) < self.TARGET_ZONE_WIDTH / 2:
                 reward += 0.01 # Small reward for aiming correctly before dropping
        
        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and self.victory:
            reward += 100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _spawn_new_block(self):
        self.block_state = BlockState.HELD
        self.block_pos = pygame.Vector2(self.arm_pos.x, self.arm_pos.y + self.ARM_HEIGHT / 2 + self.BLOCK_SIZE / 2)
        # sfx: block_spawn_sound

    def _update_arm(self, movement):
        if movement == 3:  # Left
            self.arm_pos.x -= self.ARM_SPEED
        elif movement == 4:  # Right
            self.arm_pos.x += self.ARM_SPEED
        
        arm_clamp_x = self.SCREEN_WIDTH - self.ARM_WIDTH / 2
        self.arm_pos.x = np.clip(self.arm_pos.x, self.ARM_WIDTH / 2, arm_clamp_x)
        
        if self.block_state == BlockState.HELD:
            self.block_pos.x = self.arm_pos.x

    def _handle_input(self, space_held):
        if space_held and not self.prev_space_held and self.block_state == BlockState.HELD:
            self.block_state = BlockState.FALLING
            # sfx: block_drop_sound
        self.prev_space_held = space_held

    def _update_block(self):
        if self.block_state == BlockState.FALLING:
            self.block_pos.y += self.BLOCK_FALL_SPEED

    def _update_conveyor(self):
        self.conveyor_tread_offset = (self.conveyor_tread_offset + self.conveyor_speed) % self.CONVEYOR_TREAD_SPACING
        
        for block in self.placed_blocks:
            block['pos'].x -= self.conveyor_speed
        
        self.placed_blocks = [b for b in self.placed_blocks if b['pos'].x > -self.BLOCK_SIZE]

    def _check_block_landing(self):
        reward = 0
        if self.block_state == BlockState.FALLING and self.block_pos.y >= self.CONVEYOR_Y - self.BLOCK_SIZE / 2:
            # Check for perfect placement
            target_x_on_conveyor = (self.block_pos.x - self.conveyor_tread_offset) % self.SCREEN_WIDTH
            
            # The target is a virtual 'slot' on the conveyor. We check if the block lands in it.
            # Let's define the target slot as being centered in the screen at the moment of the drop.
            # For simplicity, we'll define a static target zone on the conveyor belt itself.
            # A more dynamic approach would be to have target markers move along the belt.
            # Let's simplify: the goal is to land on *any* part of the belt. The challenge is timing.
            # The brief says "place falling blocks onto a moving conveyor belt". Let's make the target a static area.
            
            target_center_x = self.SCREEN_WIDTH / 2
            
            if abs(self.block_pos.x - target_center_x) <= self.TARGET_ZONE_WIDTH / 2:
                # --- SUCCESS ---
                self.score += 1
                reward = 1.0
                self.placed_blocks.append({'pos': self.block_pos.copy(), 'color': self.COLOR_SUCCESS})
                self._create_particles(self.block_pos, self.COLOR_SUCCESS)
                self._trigger_feedback_flash(self.COLOR_SUCCESS)
                # sfx: success_chime

                if self.score % 5 == 0 and self.score > 0:
                    self.conveyor_speed += 0.05
                
                if self.score >= self.WIN_SCORE:
                    self.victory = True
                    self.game_over = True
                else:
                    self._spawn_new_block()
            else:
                # --- FAILURE ---
                reward = -10.0
                self.game_over = True
                self.placed_blocks.append({'pos': self.block_pos.copy(), 'color': self.COLOR_FAILURE})
                self._trigger_feedback_flash(self.COLOR_FAILURE)
                # sfx: failure_buzz
            
            self.block_state = BlockState.PLACED # Block is handled, either success or fail
        return reward

    def _check_termination(self):
        return self.game_over

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "conveyor_speed": self.conveyor_speed}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    # --- RENDERING ---
    def _render_game(self):
        self._render_background_grid()
        self._render_conveyor_belt()
        self._render_placed_blocks()
        self._render_particles()
        self._render_arm()
        if self.block_state != BlockState.PLACED:
            self._render_block()
        self._render_feedback_flash_surface()

    def _render_background_grid(self):
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, (30, 35, 40), (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, (30, 35, 40), (0, i), (self.SCREEN_WIDTH, i))

    def _render_conveyor_belt(self):
        # Belt Body
        pygame.draw.rect(self.screen, self.COLOR_CONVEYOR, (0, self.CONVEYOR_Y, self.SCREEN_WIDTH, self.CONVEYOR_HEIGHT))
        
        # Target Zone
        target_rect = pygame.Rect(0,0, self.TARGET_ZONE_WIDTH, self.CONVEYOR_HEIGHT)
        target_rect.center = (self.SCREEN_WIDTH/2, self.CONVEYOR_Y + self.CONVEYOR_HEIGHT/2)
        
        s = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        s.fill(self.COLOR_TARGET_ZONE)
        self.screen.blit(s, target_rect.topleft)

        # Treads
        for i in range(-int(self.CONVEYOR_TREAD_SPACING), self.SCREEN_WIDTH, self.CONVEYOR_TREAD_SPACING):
            x = i + self.conveyor_tread_offset
            pygame.draw.line(self.screen, self.COLOR_CONVEYOR_TREADS, (x, self.CONVEYOR_Y), (x, self.CONVEYOR_Y + self.CONVEYOR_HEIGHT), 2)

    def _render_placed_blocks(self):
        for block in self.placed_blocks:
            rect = pygame.Rect(0, 0, self.BLOCK_SIZE, self.BLOCK_SIZE)
            rect.center = block['pos']
            pygame.draw.rect(self.screen, block['color'], rect, border_radius=4)
            pygame.draw.rect(self.screen, tuple(c*0.8 for c in block['color']), rect, 2, border_radius=4)

    def _render_arm(self):
        # Arm rail
        pygame.draw.rect(self.screen, (60,60,70), (0, self.arm_pos.y - 5, self.SCREEN_WIDTH, 10))
        # Arm body
        arm_rect = pygame.Rect(0, 0, self.ARM_WIDTH, self.ARM_HEIGHT)
        arm_rect.center = self.arm_pos
        
        # Glow effect
        glow_center = (int(arm_rect.centerx), int(arm_rect.centery))
        pygame.gfxdraw.filled_circle(self.screen, glow_center[0], glow_center[1], int(self.ARM_WIDTH * 0.7), self.COLOR_ARM_GLOW)
        pygame.gfxdraw.aacircle(self.screen, glow_center[0], glow_center[1], int(self.ARM_WIDTH * 0.7), self.COLOR_ARM_GLOW)

        pygame.draw.rect(self.screen, self.COLOR_ARM, arm_rect, border_radius=5)
        pygame.draw.rect(self.screen, tuple(c*0.8 for c in self.COLOR_ARM), arm_rect, 2, border_radius=5)

    def _render_block(self):
        rect = pygame.Rect(0, 0, self.BLOCK_SIZE, self.BLOCK_SIZE)
        rect.center = self.block_pos

        # Glow effect
        glow_center = (int(rect.centerx), int(rect.centery))
        pygame.gfxdraw.filled_circle(self.screen, glow_center[0], glow_center[1], int(self.BLOCK_SIZE * 1.2), self.COLOR_BLOCK_GLOW)
        pygame.gfxdraw.aacircle(self.screen, glow_center[0], glow_center[1], int(self.BLOCK_SIZE * 1.2), self.COLOR_BLOCK_GLOW)

        pygame.draw.rect(self.screen, self.COLOR_BLOCK, rect, border_radius=4)
        pygame.draw.rect(self.screen, tuple(c*0.8 for c in self.COLOR_BLOCK), rect, 2, border_radius=4)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        speed_text = self.font.render(f"SPEED: {self.conveyor_speed:.2f}", True, self.COLOR_TEXT)
        speed_rect = speed_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(speed_text, speed_rect)

    # --- VISUAL EFFECTS ---
    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(20, 41)
            self.particles.append([pos.copy(), pygame.Vector2(vx, vy), lifetime, color])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[1]  # Update position
            p[2] -= 1     # Decrease lifetime
        self.particles = [p for p in self.particles if p[2] > 0]

    def _render_particles(self):
        for p in self.particles:
            pos, _, lifetime, color = p
            alpha = max(0, 255 * (lifetime / 40))
            current_color = (*color, alpha)
            s = pygame.Surface((6, 6), pygame.SRCALPHA)
            pygame.draw.circle(s, current_color, (3, 3), int(lifetime/10))
            self.screen.blit(s, (int(pos.x - 3), int(pos.y - 3)))

    def _trigger_feedback_flash(self, color):
        self.feedback_flash['color'] = color
        self.feedback_flash['alpha'] = 128

    def _update_feedback_flash(self):
        if self.feedback_flash['alpha'] > 0:
            self.feedback_flash['alpha'] -= 8
            self.feedback_flash['alpha'] = max(0, self.feedback_flash['alpha'])

    def _render_feedback_flash_surface(self):
        if self.feedback_flash['alpha'] > 0:
            flash_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            flash_surface.fill((*self.feedback_flash['color'], self.feedback_flash['alpha']))
            self.screen.blit(flash_surface, (0, 0))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not be run by the autograder
    os.environ.pop("SDL_VIDEODRIVER", None)
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Dropper")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset() # Auto-reset for continuous play
            terminated = False
            truncated = False
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(60) # Run at 60 FPS for smooth human gameplay
        
    env.close()