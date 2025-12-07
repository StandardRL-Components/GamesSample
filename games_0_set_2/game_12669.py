import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:11:56.639952
# Source Brief: brief_02669.md
# Brief Index: 2669
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
        "Control a bouncing ball by tilting the environment. Collect the green orbs to "
        "score points and avoid the red orbs before time runs out."
    )
    user_guide = "Controls: Use the ← and → arrow keys to tilt the world and guide the ball."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    # Game rules
    FPS = 100  # Steps per second
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS
    WIN_SCORE = 400
    NUM_GREEN_ORBS = 10
    NUM_RED_ORBS = 5
    # Physics
    GRAVITY = 0.1
    TILT_FORCE = 0.2
    FRICTION = 0.99
    BOUNCE_FACTOR = 0.85
    WALL_BOUNCE_FACTOR = 0.7
    # Player
    PLAYER_RADIUS = 12
    PLAYER_TRAIL_LENGTH = 15
    # Orbs
    ORB_RADIUS = 8
    ORB_PULSE_SPEED = 0.1
    ORB_PULSE_AMP = 2
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_GOOD_ORB = (0, 255, 150)
    COLOR_BAD_ORB = (255, 50, 50)
    COLOR_UI_TEXT = (220, 220, 220)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
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
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_trail = None
        self.green_orbs = None
        self.red_orbs = None
        self.steps = 0
        self.score = 0
        self.terminated = False
        self.win = False
        
        # Call reset to set initial state
        # self.reset() # reset is called by the wrapper/runner

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = np.array([self.WIDTH / 2, self.PLAYER_RADIUS * 3], dtype=np.float32)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.player_trail = deque(maxlen=self.PLAYER_TRAIL_LENGTH)
        
        self.green_orbs = []
        self.red_orbs = []
        
        self.steps = 0
        self.score = 0
        self.terminated = False
        self.win = False

        all_orbs = []
        for _ in range(self.NUM_GREEN_ORBS):
            self._spawn_orb(self.green_orbs, all_orbs)
        for _ in range(self.NUM_RED_ORBS):
            self._spawn_orb(self.red_orbs, all_orbs)
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # 1. Handle player input
        movement = action[0]
        self._handle_input(movement)

        # 2. Update physics
        self._update_physics()

        # 3. Handle collisions and interactions
        reward += self._handle_collisions()

        # 4. Check for termination conditions
        time_out = self.steps >= self.MAX_STEPS
        self.win = self.score >= self.WIN_SCORE
        
        if self.win:
            reward += 100 # Goal-oriented reward for winning
            self.terminated = True
        elif time_out:
            self.terminated = True

        # Truncated is not used in this environment
        truncated = False

        return (
            self._get_observation(),
            reward,
            self.terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement):
        # 3=left, 4=right
        if movement == 3:
            self.player_vel[0] -= self.TILT_FORCE
        elif movement == 4:
            self.player_vel[0] += self.TILT_FORCE

    def _update_physics(self):
        # Apply gravity
        self.player_vel[1] += self.GRAVITY
        # Apply friction
        self.player_vel[0] *= self.FRICTION
        
        # Update position
        self.player_pos += self.player_vel
        
        # Update trail
        if self.steps % 2 == 0: # Add to trail every other step
             self.player_trail.append(self.player_pos.copy())

    def _handle_collisions(self):
        step_reward = 0
        
        # Wall collisions (left/right)
        if self.player_pos[0] < self.PLAYER_RADIUS:
            self.player_pos[0] = self.PLAYER_RADIUS
            self.player_vel[0] *= -self.WALL_BOUNCE_FACTOR
            self.score = int(self.score * 0.75) # Score penalty
            step_reward -= 2.5 # RL reward penalty
            # sfx: wall_thud
        elif self.player_pos[0] > self.WIDTH - self.PLAYER_RADIUS:
            self.player_pos[0] = self.WIDTH - self.PLAYER_RADIUS
            self.player_vel[0] *= -self.WALL_BOUNCE_FACTOR
            self.score = int(self.score * 0.75) # Score penalty
            step_reward -= 2.5 # RL reward penalty
            # sfx: wall_thud
            
        # Floor/Ceiling collisions
        if self.player_pos[1] > self.HEIGHT - self.PLAYER_RADIUS:
            self.player_pos[1] = self.HEIGHT - self.PLAYER_RADIUS
            self.player_vel[1] *= -self.BOUNCE_FACTOR
            # sfx: bounce
        elif self.player_pos[1] < self.PLAYER_RADIUS:
            self.player_pos[1] = self.PLAYER_RADIUS
            self.player_vel[1] *= -self.BOUNCE_FACTOR
            # sfx: bounce
            
        # Orb collisions
        all_orbs = [(self.green_orbs, True), (self.red_orbs, False)]
        for orb_list, is_green in all_orbs:
            for i, orb_pos in reversed(list(enumerate(orb_list))):
                dist = np.linalg.norm(self.player_pos - orb_pos)
                if dist < self.PLAYER_RADIUS + self.ORB_RADIUS:
                    if is_green:
                        self.score += 10
                        step_reward += 1
                        # sfx: collect_good
                    else:
                        self.score -= 20
                        step_reward -= 2
                        # sfx: collect_bad
                    
                    orb_list.pop(i)
                    self._spawn_orb(orb_list, self.green_orbs + self.red_orbs)
        
        return step_reward

    def _spawn_orb(self, orb_list_to_add, all_orbs):
        while True:
            pos = np.array([
                self.np_random.uniform(self.ORB_RADIUS, self.WIDTH - self.ORB_RADIUS),
                self.np_random.uniform(self.ORB_RADIUS, self.HEIGHT - self.ORB_RADIUS)
            ])
            # Ensure not too close to other orbs or the player
            valid_pos = True
            if self.player_pos is not None and np.linalg.norm(pos - self.player_pos) < self.PLAYER_RADIUS * 5:
                valid_pos = False
            if valid_pos:
                for other_orb in all_orbs:
                    if np.linalg.norm(pos - other_orb) < self.ORB_RADIUS * 4:
                        valid_pos = False
                        break
            if valid_pos:
                orb_list_to_add.append(pos)
                break

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render orbs first
        self._render_orbs(self.green_orbs, self.COLOR_GOOD_ORB)
        self._render_orbs(self.red_orbs, self.COLOR_BAD_ORB)
        
        # Render player trail
        if self.player_trail:
            self._render_player_trail()
        
        # Render player
        if self.player_pos is not None:
            self._render_player()
        
        # Render win/lose message
        if self.terminated:
            message = "YOU WIN!" if self.win else "TIME UP!"
            color = self.COLOR_GOOD_ORB if self.win else self.COLOR_BAD_ORB
            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def _render_orbs(self, orbs, color):
        pulse = math.sin(self.steps * self.ORB_PULSE_SPEED) * self.ORB_PULSE_AMP
        radius = self.ORB_RADIUS + pulse
        
        for orb_pos in orbs:
            x, y = int(orb_pos[0]), int(orb_pos[1])
            # Glow effect
            glow_radius = int(radius * 1.8)
            glow_color = tuple(c // 3 for c in color)
            pygame.gfxdraw.filled_circle(self.screen, x, y, glow_radius, glow_color)
            pygame.gfxdraw.aacircle(self.screen, x, y, glow_radius, glow_color)
            
            # Main orb
            pygame.gfxdraw.filled_circle(self.screen, x, y, int(radius), color)
            pygame.gfxdraw.aacircle(self.screen, x, y, int(radius), color)

    def _render_player_trail(self):
        for i, pos in enumerate(self.player_trail):
            alpha = int(255 * (i / self.PLAYER_TRAIL_LENGTH))
            color = (self.COLOR_PLAYER[0], self.COLOR_PLAYER[1], self.COLOR_PLAYER[2], alpha // 4)
            radius = int(self.PLAYER_RADIUS * (i / self.PLAYER_TRAIL_LENGTH))
            if radius > 0:
                # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, color)
                self.screen.blit(temp_surf, (int(pos[0]) - radius, int(pos[1]) - radius))


    def _render_player(self):
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        
        # Glow effect
        for i in range(4, 0, -1):
            radius = self.PLAYER_RADIUS + i * 3
            alpha = 80 - i * 20
            # Create a temporary surface for alpha blending
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, (*self.COLOR_PLAYER, alpha))
            self.screen.blit(temp_surf, (x - radius, y - radius))

        # Main player circle
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, x, y, self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        time_left = max(0, self.GAME_DURATION_SECONDS - (self.steps / self.FPS))
        timer_text = self.font_small.render(f"TIME: {time_left:.2f}", True, self.COLOR_UI_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, self.GAME_DURATION_SECONDS - (self.steps / self.FPS))
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # --- Example Usage & Human Play ---
    # This part is for testing and will not be part of the final environment.
    # It requires a display.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    
    # Setup for human play
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Bounce & Collect")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    print("\n--- Human Controls ---")
    print("Left Arrow: Tilt Left")
    print("Right Arrow: Tilt Right")
    print("R: Reset Environment")
    print("Q: Quit")
    print("----------------------\n")

    running = True
    while running:
        # Action defaults to no-op
        action = np.array([0, 0, 0]) 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_q:
                    running = False

        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(60) # Limit human play to 60 FPS for playability

    env.close()
    pygame.quit()