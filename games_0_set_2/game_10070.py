import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:52:11.977245
# Source Brief: brief_00070.md
# Brief Index: 70
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a bouncing ball to collect
    glowing orbs within a time limit. The core mechanic involves managing
    momentum and bounce height, which increases with each orb collected.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a bouncing ball to collect glowing orbs against the clock. "
        "Each orb increases your bounciness, helping you reach higher."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to aim. Press spacebar while on the ground to launch. "
        "The longer you wait on the ground, the more powerful your launch will be."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30  # Simulation FPS
    GAME_DURATION_SECONDS = 34  # ~1000 steps at 30 FPS
    MAX_STEPS = int(GAME_DURATION_SECONDS * FPS)

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_BALL = (60, 180, 255)
    COLOR_BALL_GLOW = (60, 180, 255)
    COLOR_ORB = (255, 165, 0)
    COLOR_ORB_GLOW = (255, 165, 0)
    COLOR_WALL = (200, 200, 220)
    COLOR_TRAIL = (100, 200, 255)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_AIM_INDICATOR = (255, 255, 255)
    
    # Game Parameters
    TARGET_ORBS = 30
    BALL_RADIUS = 12
    ORB_RADIUS = 8
    GRAVITY = 0.45
    BASE_RESTITUTION = 0.85  # Bounciness
    LAUNCH_POWER_BASE = 8.0
    LAUNCH_POWER_SCALAR = 10.0 # Power per second spent on ground
    MAX_LAUNCH_POWER = 25.0
    TRAIL_LENGTH = 20

    # Reward Parameters
    REWARD_ORB_COLLECT = 1.0
    REWARD_WIN = 100.0
    REWARD_LOSE = -10.0
    REWARD_WALL_HIT = -0.01
    REWARD_RECURSIVE_BOUNCE = 0.5
    REWARD_CLOSER_TO_ORB_SCALAR = 0.2

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
        try:
            self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 24)

        self.render_mode = render_mode
        self._initialize_state_variables()
        # self.validate_implementation() # Removed for submission

    def _initialize_state_variables(self):
        """Initializes all state variables to default values."""
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_left = self.GAME_DURATION_SECONDS
        
        self.ball_pos = np.zeros(2, dtype=np.float32)
        self.ball_vel = np.zeros(2, dtype=np.float32)
        self.orbs_collected = 0
        self.restitution = self.BASE_RESTITUTION
        
        self.orbs = []
        self.particles = []
        self.trail = deque(maxlen=self.TRAIL_LENGTH)
        
        self.is_on_ground = False
        self.time_on_ground = 0.0
        self.launch_direction = 1 # 1: up
        self.dist_to_nearest_orb = float('inf')
        self.bounce_history = deque(maxlen=2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state_variables()

        self.ball_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - self.BALL_RADIUS - 1.0], dtype=np.float32)
        self.is_on_ground = True
        
        self._generate_orbs()
        self.dist_to_nearest_orb = self._get_dist_to_nearest_orb()
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action
        reward = 0.0
        
        dist_before = self._get_dist_to_nearest_orb()
        
        # 1. Handle Input
        self._handle_input(movement, space_held == 1)

        # 2. Update Physics & Game State
        self._update_physics_and_state()

        # 3. Handle Collisions and Collect Rewards
        collision_reward, orbs_hit = self._handle_collisions()
        reward += collision_reward

        # 4. Calculate Shaping Rewards
        dist_after = self._get_dist_to_nearest_orb()
        if dist_after < dist_before:
            reward += (dist_before - dist_after) * self.REWARD_CLOSER_TO_ORB_SCALAR
        
        self.dist_to_nearest_orb = dist_after
        self.score += reward

        # 5. Check Termination
        terminated = self._check_termination()
        truncated = False # No truncation condition other than termination
        if terminated:
            self.game_over = True
            if self.orbs_collected >= self.TARGET_ORBS:
                reward += self.REWARD_WIN
                self.score += self.REWARD_WIN
            else:
                reward += self.REWARD_LOSE
                self.score += self.REWARD_LOSE

        obs = self._get_observation()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _handle_input(self, movement, space_held):
        if movement in [1, 2, 3, 4]:
            self.launch_direction = movement

        if space_held and self.is_on_ground:
            # Calculate launch power based on time on ground
            charge_time = self.time_on_ground / self.FPS
            power = self.LAUNCH_POWER_BASE + charge_time * self.LAUNCH_POWER_SCALAR
            power = min(power, self.MAX_LAUNCH_POWER)
            
            # Apply launch velocity
            if self.launch_direction == 1: # Up
                self.ball_vel[1] = -power
            elif self.launch_direction == 2: # Down (less useful, but for completeness)
                self.ball_vel[1] = power / 2 # Pushing into ground is weak
            elif self.launch_direction == 3: # Left
                self.ball_vel[0] = -power
            elif self.launch_direction == 4: # Right
                self.ball_vel[0] = power

            self.is_on_ground = False
            self.time_on_ground = 0
            # sfx: launch_sound

    def _update_physics_and_state(self):
        self.steps += 1
        self.time_left -= 1.0 / self.FPS

        if self.is_on_ground:
            self.time_on_ground += 1
            self.ball_vel[0] *= 0.9 # Ground friction
        else:
            # Apply gravity
            self.ball_vel[1] += self.GRAVITY
        
        # Update position
        self.ball_pos += self.ball_vel

        # Update trail
        self.trail.append(self.ball_pos.copy())

        # Update particles
        self.particles = [p for p in self.particles if self._update_particle(p)]

    def _handle_collisions(self):
        reward = 0.0
        orbs_hit_this_step = 0
        
        # Wall collisions
        if self.ball_pos[0] < self.BALL_RADIUS:
            self.ball_pos[0] = self.BALL_RADIUS
            self.ball_vel[0] *= -self.restitution
            reward += self.REWARD_WALL_HIT
        elif self.ball_pos[0] > self.SCREEN_WIDTH - self.BALL_RADIUS:
            self.ball_pos[0] = self.SCREEN_WIDTH - self.BALL_RADIUS
            self.ball_vel[0] *= -self.restitution
            reward += self.REWARD_WALL_HIT

        if self.ball_pos[1] < self.BALL_RADIUS:
            self.ball_pos[1] = self.BALL_RADIUS
            self.ball_vel[1] *= -self.restitution
            reward += self.REWARD_WALL_HIT
        elif self.ball_pos[1] > self.SCREEN_HEIGHT - self.BALL_RADIUS:
            self.ball_pos[1] = self.SCREEN_HEIGHT - self.BALL_RADIUS
            self.ball_vel[1] *= -self.restitution
            self.is_on_ground = True
            reward += self.REWARD_WALL_HIT
            # sfx: bounce_sound

            # Recursive bounce check
            bounce_data = {'pos': self.ball_pos.copy(), 'time': self.steps}
            if len(self.bounce_history) == 2:
                last_bounce = self.bounce_history[1]
                dist = np.linalg.norm(bounce_data['pos'] - last_bounce['pos'])
                time_diff = bounce_data['time'] - last_bounce['time']
                if dist < 32 and time_diff < self.FPS * 1.5: # 32px is ~5% of width
                    reward += self.REWARD_RECURSIVE_BOUNCE
            self.bounce_history.append(bounce_data)

        # Orb collisions
        remaining_orbs = []
        for orb_pos in self.orbs:
            dist = np.linalg.norm(self.ball_pos - orb_pos)
            if dist < self.BALL_RADIUS + self.ORB_RADIUS:
                self.orbs_collected += 1
                orbs_hit_this_step += 1
                reward += self.REWARD_ORB_COLLECT
                # sfx: orb_collect_sound
                self._create_particles(orb_pos, self.COLOR_ORB, 20)
                # Increase restitution (bounciness)
                self.restitution = self.BASE_RESTITUTION * (1 + 0.05 * self.orbs_collected)
                self.restitution = min(self.restitution, 0.98) # Cap to prevent energy gain
            else:
                remaining_orbs.append(orb_pos)
        self.orbs = remaining_orbs
        
        return reward, orbs_hit_this_step

    def _check_termination(self):
        return (self.orbs_collected >= self.TARGET_ORBS or 
                self.time_left <= 0 or 
                self.steps >= self.MAX_STEPS)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 1)

        # Draw trail
        if len(self.trail) > 1:
            for i in range(len(self.trail) - 1):
                alpha = int(255 * (i / self.TRAIL_LENGTH))
                color_with_alpha = self.COLOR_TRAIL + (alpha,)
                try: # For compatibility with surfaces that don't support alpha
                    pygame.draw.line(
                        self.screen, color_with_alpha, 
                        self.trail[i].astype(int), self.trail[i+1].astype(int), 
                        width=max(1, int(self.BALL_RADIUS * 0.5 * (i / self.TRAIL_LENGTH)))
                    )
                except (ValueError, TypeError): # Fallback for older pygame or no alpha
                     pygame.draw.line(
                        self.screen, self.COLOR_TRAIL, 
                        self.trail[i].astype(int), self.trail[i+1].astype(int), 
                        width=max(1, int(self.BALL_RADIUS * 0.5 * (i / self.TRAIL_LENGTH)))
                    )

        # Draw orbs
        pulse = math.sin(self.steps * 0.2) * 2
        for orb_pos in self.orbs:
            self._draw_glowing_circle(self.screen, orb_pos.astype(int), self.ORB_RADIUS + pulse, self.COLOR_ORB, self.COLOR_ORB_GLOW)

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'].astype(int), int(p['radius']))

        # Draw ball
        self._draw_glowing_circle(self.screen, self.ball_pos.astype(int), self.BALL_RADIUS, self.COLOR_BALL, self.COLOR_BALL_GLOW)
        
        # Draw aim indicator
        if self.is_on_ground:
            end_pos = self.ball_pos.copy()
            length = 20 + self.time_on_ground * 0.5
            if self.launch_direction == 1: end_pos[1] -= length
            elif self.launch_direction == 2: end_pos[1] += length
            elif self.launch_direction == 3: end_pos[0] -= length
            elif self.launch_direction == 4: end_pos[0] += length
            
            alpha = min(255, 100 + int(self.time_on_ground * 3))
            color_with_alpha = self.COLOR_AIM_INDICATOR + (alpha,)
            try:
                pygame.draw.line(self.screen, color_with_alpha, self.ball_pos.astype(int), end_pos.astype(int), 2)
            except (ValueError, TypeError):
                pygame.draw.line(self.screen, self.COLOR_AIM_INDICATOR, self.ball_pos.astype(int), end_pos.astype(int), 2)

    def _render_ui(self):
        time_text = f"TIME: {max(0, self.time_left):.1f}"
        orbs_text = f"ORBS: {self.orbs_collected}/{self.TARGET_ORBS}"
        
        time_surf = self.font_large.render(time_text, True, self.COLOR_UI_TEXT)
        orbs_surf = self.font_small.render(orbs_text, True, self.COLOR_UI_TEXT)

        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))
        self.screen.blit(orbs_surf, (self.SCREEN_WIDTH - orbs_surf.get_width() - 10, 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "orbs_collected": self.orbs_collected,
            "time_left": self.time_left
        }

    def _generate_orbs(self):
        self.orbs = []
        min_dist = 2 * self.ORB_RADIUS
        for _ in range(self.TARGET_ORBS):
            while True:
                pos = np.array([
                    self.np_random.uniform(self.ORB_RADIUS * 2, self.SCREEN_WIDTH - self.ORB_RADIUS * 2),
                    self.np_random.uniform(self.ORB_RADIUS * 2, self.SCREEN_HEIGHT - self.ORB_RADIUS * 4) # Keep away from floor
                ])
                if np.linalg.norm(pos - self.ball_pos) < 100: continue # Don't spawn too close to start
                if all(np.linalg.norm(pos - other_pos) > min_dist for other_pos in self.orbs):
                    self.orbs.append(pos)
                    break
    
    def _get_dist_to_nearest_orb(self):
        if not self.orbs:
            return 0
        distances = [np.linalg.norm(self.ball_pos - orb_pos) for orb_pos in self.orbs]
        return min(distances)

    def _draw_glowing_circle(self, surface, pos, radius, color, glow_color):
        # Simple glow effect by drawing larger, semi-transparent circles
        for i in range(4, 0, -1):
            alpha = 40 // (i * 2)
            try:
                pygame.gfxdraw.filled_circle(
                    surface, pos[0], pos[1], 
                    int(radius + i * 2), 
                    (glow_color[0], glow_color[1], glow_color[2], alpha)
                )
            except (ValueError, TypeError): # Fallback for surfaces that don't support alpha
                pass # Just skip glow if not supported
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(radius), color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(radius), color)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'lifespan': self.np_random.integers(10, 21),
                'color': color + (255,)
            })

    def _update_particle(self, p):
        p['pos'] += p['vel']
        p['lifespan'] -= 1
        p['radius'] *= 0.95
        alpha = int(255 * (p['lifespan'] / 20))
        p['color'] = (p['color'][0], p['color'][1], p['color'][2], max(0, alpha))
        return p['lifespan'] > 0 and p['radius'] > 0.5

    def close(self):
        pygame.font.quit()
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bouncing Orb Collector")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    # Action state
    movement = 0
    space_held = 0
    
    print("--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    terminated = True
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                if event.key == pygame.K_SPACE: space_held = 1
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    # In a real game, you might want to reset movement to 0 here
                    # But for this action scheme, we keep the last direction.
                    pass
                if event.key == pygame.K_SPACE: space_held = 0
                
        action = [movement, space_held, 0] # Shift is not used
        
        obs, reward, term, trunc, info = env.step(action)
        terminated = term or trunc
        total_reward += reward
        
        # Draw the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Reset one-shot actions
        # movement = 0 # Keep direction until changed

        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {info['score']:.2f}, Orbs: {info['orbs_collected']}")
    env.close()