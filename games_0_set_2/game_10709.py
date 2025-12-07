import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:52:05.493489
# Source Brief: brief_00709.md
# Brief Index: 709
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque

class GameEnv(gym.Env):
    """
    Gymnasium environment for a game where the player controls a bouncing ball
    to hit targets. Hitting targets increases score and ball speed. Chaining
    hits builds a combo for bonus rewards. The goal is to reach 1000 points
    within 60 seconds.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a bouncing ball to hit targets. Chaining hits builds a combo for bonus rewards and increases ball speed."
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to nudge the ball and hit all the targets before time runs out."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60  # For smooth visuals and physics updates
        self.MAX_STEPS = 60 * self.FPS  # 60-second game
        self.WIN_SCORE = 1000

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_BALL = (255, 60, 60)
        self.COLOR_BALL_GLOW = (255, 100, 100)
        self.COLOR_TARGET = (240, 240, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HIT_FLASH = (100, 255, 100)
        self.COLOR_COMBO_FLASH = (255, 255, 100)
        
        # Game Mechanics
        self.BALL_RADIUS = 12
        self.BALL_INITIAL_SPEED = 4
        self.BALL_SPEED_INCREASE = 1.05  # 5% increase
        self.HORIZONTAL_NUDGE = 0.4
        self.MAX_HORIZONTAL_SPEED = 7

        self.TARGET_COUNT = 10
        self.TARGET_WIDTH, self.TARGET_HEIGHT = 50, 20
        self.TARGET_RESPAWN_DELAY = int(0.5 * self.FPS) # in frames

        self.COMBO_THRESHOLD = 5
        self.TRAIL_LENGTH = 15

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 24)
        self.font_small = pygame.font.SysFont("Arial", 18)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ball_pos = [0, 0]
        self.ball_vel = [0, 0]
        self.targets = []
        self.combo_hits = 0
        self.effects = []
        self.ball_trail = deque(maxlen=self.TRAIL_LENGTH)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.combo_hits = 0
        self.effects = []

        # Initialize ball
        self.ball_pos = [self.WIDTH / 2, self.HEIGHT / 2]
        angle = self.np_random.uniform(0, 2 * math.pi)
        self.ball_vel = [
            self.BALL_INITIAL_SPEED * math.cos(angle),
            self.BALL_INITIAL_SPEED * math.sin(angle),
        ]
        
        self.ball_trail.clear()
        for _ in range(self.TRAIL_LENGTH):
            self.ball_trail.append(list(self.ball_pos))

        # Initialize targets
        self.targets = []
        for _ in range(self.TARGET_COUNT):
            self.targets.append(self._create_new_target())

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        
        # --- 1. Handle Action ---
        movement = action[0]
        if movement == 3:  # Left
            self.ball_vel[0] -= self.HORIZONTAL_NUDGE
        elif movement == 4:  # Right
            self.ball_vel[0] += self.HORIZONTAL_NUDGE
        
        # Clamp horizontal speed
        self.ball_vel[0] = np.clip(self.ball_vel[0], -self.MAX_HORIZONTAL_SPEED, self.MAX_HORIZONTAL_SPEED)

        # --- 2. Update Game State ---
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        self.ball_trail.append(list(self.ball_pos))
        
        self._handle_wall_collisions()
        reward += self._handle_target_collisions()
        self._update_targets()
        self._update_effects()

        self.steps += 1

        # --- 3. Check Termination ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100.0
            terminated = True
            # sfx: win_sound
        elif self.steps >= self.MAX_STEPS:
            reward -= 10.0
            terminated = True
            # sfx: lose_sound

        self.game_over = terminated
        
        truncated = False # This environment does not truncate based on time limit

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_wall_collisions(self):
        # Left/Right walls
        if self.ball_pos[0] <= self.BALL_RADIUS:
            self.ball_pos[0] = self.BALL_RADIUS
            self.ball_vel[0] *= -1
            # sfx: wall_bounce
        elif self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
            self.ball_pos[0] = self.WIDTH - self.BALL_RADIUS
            self.ball_vel[0] *= -1
            # sfx: wall_bounce

        # Top/Bottom walls
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_pos[1] = self.BALL_RADIUS
            self.ball_vel[1] *= -1
            # sfx: wall_bounce
        elif self.ball_pos[1] >= self.HEIGHT - self.BALL_RADIUS:
            self.ball_pos[1] = self.HEIGHT - self.BALL_RADIUS
            self.ball_vel[1] *= -1
            # sfx: wall_bounce
            
    def _handle_target_collisions(self):
        hit_reward = 0.0
        ball_rect = pygame.Rect(
            self.ball_pos[0] - self.BALL_RADIUS,
            self.ball_pos[1] - self.BALL_RADIUS,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
        )

        for target in self.targets:
            if target['respawn_timer'] > 0:
                continue

            if ball_rect.colliderect(target['rect']):
                # sfx: target_hit
                self.score += 10
                hit_reward += 0.1
                
                # Deactivate target
                target['respawn_timer'] = self.TARGET_RESPAWN_DELAY
                
                # Add hit effect
                self.effects.append({
                    'type': 'flash', 'pos': target['rect'].center,
                    'color': self.COLOR_HIT_FLASH, 'max_radius': 40,
                    'timer': 10, 'max_timer': 10
                })

                # Combo logic
                self.combo_hits += 1
                if self.combo_hits > 0 and self.combo_hits % self.COMBO_THRESHOLD == 0:
                    hit_reward += 1.0
                    # sfx: combo_achieved
                    self.effects.append({
                        'type': 'flash', 'pos': self.ball_pos,
                        'color': self.COLOR_COMBO_FLASH, 'max_radius': 80,
                        'timer': 20, 'max_timer': 20
                    })
                
                # Increase ball speed
                speed = math.hypot(*self.ball_vel)
                if speed > 0: # Avoid division by zero
                    new_speed = speed * self.BALL_SPEED_INCREASE
                    self.ball_vel = [v * (new_speed / speed) for v in self.ball_vel]

                # Bounce logic
                offset_x = self.ball_pos[0] - target['rect'].centerx
                offset_y = self.ball_pos[1] - target['rect'].centery
                
                if abs(offset_x) > abs(offset_y): # Side hit
                    self.ball_vel[0] *= -1
                else: # Top/bottom hit
                    self.ball_vel[1] *= -1
                
                # Nudge ball out of target to prevent sticking
                self.ball_pos[0] += self.ball_vel[0]
                self.ball_pos[1] += self.ball_vel[1]

                break # Only one collision per frame
        
        return hit_reward

    def _update_targets(self):
        for i, target in enumerate(self.targets):
            if target['respawn_timer'] > 0:
                target['respawn_timer'] -= 1
                if target['respawn_timer'] <= 0:
                    self.targets[i] = self._create_new_target()

    def _create_new_target(self):
        return {
            'rect': pygame.Rect(
                self.np_random.integers(0, self.WIDTH - self.TARGET_WIDTH),
                self.np_random.integers(50, self.HEIGHT - self.TARGET_HEIGHT - 50),
                self.TARGET_WIDTH,
                self.TARGET_HEIGHT,
            ),
            'respawn_timer': 0
        }

    def _update_effects(self):
        self.effects = [e for e in self.effects if e['timer'] > 0]
        for effect in self.effects:
            effect['timer'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Render targets
        for target in self.targets:
            if target['respawn_timer'] == 0:
                pygame.draw.rect(self.screen, self.COLOR_TARGET, target['rect'], border_radius=3)
        
        # Render trail
        for i, pos in enumerate(self.ball_trail):
            alpha = int(255 * (i / self.TRAIL_LENGTH) * 0.5)
            if alpha > 0:
                trail_color = self.COLOR_BALL_GLOW + (alpha,)
                pygame.gfxdraw.filled_circle(
                    self.screen, int(pos[0]), int(pos[1]),
                    int(self.BALL_RADIUS * (i / self.TRAIL_LENGTH)),
                    trail_color
                )

        # Render ball glow
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_BALL_GLOW + (80,), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (int(self.ball_pos[0] - glow_radius), int(self.ball_pos[1] - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

        # Render ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)

        # Render effects
        for effect in self.effects:
            if effect['type'] == 'flash':
                progress = 1.0 - (effect['timer'] / effect['max_timer'])
                radius = int(effect['max_radius'] * progress)
                alpha = int(255 * (1.0 - progress))
                if alpha > 0:
                    color = effect['color'] + (alpha,)
                    pygame.gfxdraw.filled_circle(self.screen, int(effect['pos'][0]), int(effect['pos'][1]), radius, color)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_medium.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Combo
        if self.combo_hits > 1:
            combo_text = self.font_small.render(f"COMBO x{self.combo_hits}", True, self.COLOR_COMBO_FLASH)
            text_rect = combo_text.get_rect(center=(self.WIDTH / 2, 25))
            self.screen.blit(combo_text, text_rect)
            
        # Game Over Message
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
                color = self.COLOR_COMBO_FLASH
            else:
                msg = "TIME'S UP!"
                color = self.COLOR_BALL
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()
        
    def render(self):
        # This method is not used in the standard gym loop but is useful for human play
        return self._get_observation()


if __name__ == '__main__':
    # --- Human Play Example ---
    # Re-enable display for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Bouncy Ball Target Practice")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        if terminated:
            # Wait for a moment on the game over screen, then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

        # --- Event Handling ---
        action_to_take = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    terminated = False

        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        # The original code had up/down mapped, but they don't do anything in step()
        # movement = 1 is UP, movement = 2 is DOWN
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Rendering ---
        # The observation is already the rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Clock ---
        clock.tick(env.FPS)
        
    env.close()