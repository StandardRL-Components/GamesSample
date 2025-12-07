import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:21:51.221711
# Source Brief: brief_02127.md
# Brief Index: 2127
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
        "Control a bouncing ball to collect orbs for points and speed boosts, "
        "while avoiding static obstacles and managing your momentum."
    )
    user_guide = (
        "Use the ← and → arrow keys to tilt the ground and guide the bouncing ball. "
        "Collect green orbs for points and speed, and red orbs for more points but a speed penalty."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_TOP = (15, 25, 40)
    COLOR_BG_BOTTOM = (30, 50, 80)
    COLOR_PLAYER = (255, 200, 0)
    COLOR_PLAYER_GLOW = (255, 200, 0, 50)
    COLOR_GREEN_ORB = (0, 255, 150)
    COLOR_RED_ORB = (255, 50, 100)
    COLOR_OBSTACLE = (50, 60, 70)
    COLOR_OBSTACLE_OUTLINE = (80, 90, 100)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_UI_SHADOW = (20, 20, 30)
    COLOR_HEART = (255, 80, 80)

    # Physics
    GRAVITY = 0.2
    TILT_FORCE = 0.35
    FRICTION = 0.98
    BOUNCE_DAMPING = 0.8
    MAX_VELOCITY_X = 8

    # Game Parameters
    PLAYER_RADIUS = 15
    ORB_RADIUS = 8
    INITIAL_LIVES = 3
    WIN_GREEN_ORBS = 10
    WIN_RED_ORBS = 5
    LEVEL_UP_ORB_COUNT = 5
    SPEED_LOSS_THRESHOLD = 0.2
    MAX_STEPS = 1500

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
            self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_game_over = pygame.font.SysFont(None, 60)

        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.orbs = []
        self.obstacles = []
        self.particles = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        
        self.lives = self.INITIAL_LIVES
        self.speed_multiplier = 1.0
        self.green_orbs_collected = 0
        self.red_orbs_collected = 0
        self.level = 1

        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.PLAYER_RADIUS + 10)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.invincibility_timer = 0
        
        self.orbs.clear()
        self.obstacles.clear()
        self.particles.clear()
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        reward = 0.0
        
        self._update_physics(movement)
        
        # Continuous rewards
        if self.speed_multiplier > 0.5: reward += 0.01
        if self.speed_multiplier < 1.0: reward -= 0.01

        # Event-based rewards from collisions
        reward += self._handle_collisions()
        
        self._update_particles()

        self.steps += 1
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1
        
        # Check for speed-based life loss
        if self.speed_multiplier < self.SPEED_LOSS_THRESHOLD and self.lives > 0:
            self.lives -= 1
            self.speed_multiplier = 0.5 # Recover to a safe speed
            self.invincibility_timer = 60 # Brief invincibility
            reward -= 10.0
            # Sound: Player_Hurt_LowSpeed.wav
            
        terminated = self._check_termination()

        if terminated and not self.game_over:
            self.game_over = True
            if self.win_condition_met:
                reward += 50.0 # Win bonus
            elif self.lives <= 0:
                reward -= 50.0 # Loss penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_physics(self, movement):
        # Apply gravity
        self.player_vel.y += self.GRAVITY

        # Apply tilt force only when on ground
        if self.on_ground:
            if movement == 3: # Left
                self.player_vel.x -= self.TILT_FORCE
            elif movement == 4: # Right
                self.player_vel.x += self.TILT_FORCE
        
        # Apply friction
        self.player_vel.x *= self.FRICTION
        
        # Clamp horizontal velocity
        self.player_vel.x = max(-self.MAX_VELOCITY_X, min(self.MAX_VELOCITY_X, self.player_vel.x))

        # Update position
        self.player_pos += self.player_vel * max(0.1, self.speed_multiplier) # Ensure minimum movement

        # Boundary checks and bouncing
        self.on_ground = False
        if self.player_pos.y + self.PLAYER_RADIUS >= self.SCREEN_HEIGHT:
            self.player_pos.y = self.SCREEN_HEIGHT - self.PLAYER_RADIUS
            self.player_vel.y *= -self.BOUNCE_DAMPING
            self.on_ground = True
            # Sound: Ball_Bounce.wav
        
        if self.player_pos.y - self.PLAYER_RADIUS <= 0:
            self.player_pos.y = self.PLAYER_RADIUS
            self.player_vel.y *= -self.BOUNCE_DAMPING

        if self.player_pos.x + self.PLAYER_RADIUS >= self.SCREEN_WIDTH:
            self.player_pos.x = self.SCREEN_WIDTH - self.PLAYER_RADIUS
            self.player_vel.x *= -1
        
        if self.player_pos.x - self.PLAYER_RADIUS <= 0:
            self.player_pos.x = self.PLAYER_RADIUS
            self.player_vel.x *= -1

    def _handle_collisions(self):
        reward = 0.0
        
        # Orb collisions
        for orb in self.orbs[:]:
            dist = self.player_pos.distance_to(orb['pos'])
            if dist < self.PLAYER_RADIUS + self.ORB_RADIUS:
                if orb['type'] == 'green':
                    self.score += 1
                    self.green_orbs_collected += 1
                    self.speed_multiplier += 0.05
                    reward += 1.0
                    # Sound: Collect_Green.wav
                else: # Red
                    self.score += 3
                    self.red_orbs_collected += 1
                    self.speed_multiplier -= 0.10
                    reward += 3.0
                    # Sound: Collect_Red.wav
                
                self.speed_multiplier = max(0, min(2.0, self.speed_multiplier))
                self._create_particle_burst(orb['pos'], orb['color'])
                self.orbs.remove(orb)
                self._spawn_orb()

                # Level up check
                total_orbs = self.green_orbs_collected + self.red_orbs_collected
                if total_orbs >= self.level * self.LEVEL_UP_ORB_COUNT:
                    self.level += 1
                    self._add_obstacles(2)
                    # Sound: Level_Up.wav

        # Obstacle collisions
        player_rect = pygame.Rect(
            self.player_pos.x - self.PLAYER_RADIUS,
            self.player_pos.y - self.PLAYER_RADIUS,
            self.PLAYER_RADIUS * 2,
            self.PLAYER_RADIUS * 2
        )
        if self.invincibility_timer == 0:
            for obstacle in self.obstacles:
                if player_rect.colliderect(obstacle):
                    self.lives -= 1
                    self.invincibility_timer = 90 # 1.5 seconds of invincibility
                    reward -= 10.0
                    # Sound: Player_Hurt_Obstacle.wav
                    
                    # Knockback
                    self.player_vel *= -0.75
                    break # Only one collision per frame
        
        return reward

    def _check_termination(self):
        win_by_green = self.green_orbs_collected >= self.WIN_GREEN_ORBS
        win_by_red = self.red_orbs_collected >= self.WIN_RED_ORBS
        
        if win_by_green or win_by_red:
            self.win_condition_met = True
            return True
            
        if self.lives <= 0:
            return True
            
        if self.steps >= self.MAX_STEPS:
            return True
            
        return False

    def _generate_level(self):
        self.obstacles.clear()
        self._add_obstacles(2)
        
        self.orbs.clear()
        for _ in range(5):
            self._spawn_orb()

    def _add_obstacles(self, count):
        for _ in range(count):
            while True:
                w = self.np_random.integers(60, 150)
                h = self.np_random.integers(15, 25)
                x = self.np_random.integers(0, self.SCREEN_WIDTH - w)
                y = self.np_random.integers(100, self.SCREEN_HEIGHT - h - 50)
                new_obstacle = pygame.Rect(x, y, w, h)
                
                # Ensure it doesn't overlap with existing obstacles
                if not any(new_obstacle.colliderect(obs) for obs in self.obstacles):
                    self.obstacles.append(new_obstacle)
                    break
    
    def _spawn_orb(self):
        orb_type = 'green' if self.np_random.random() > 0.4 else 'red'
        color = self.COLOR_GREEN_ORB if orb_type == 'green' else self.COLOR_RED_ORB
        
        for _ in range(10): # Try 10 times to find a good spot
            pos = pygame.Vector2(
                self.np_random.integers(self.ORB_RADIUS, self.SCREEN_WIDTH - self.ORB_RADIUS),
                self.np_random.integers(self.ORB_RADIUS, self.SCREEN_HEIGHT - self.ORB_RADIUS - 40)
            )
            
            # Check for overlap with obstacles
            orb_rect = pygame.Rect(pos.x - self.ORB_RADIUS, pos.y - self.ORB_RADIUS, self.ORB_RADIUS*2, self.ORB_RADIUS*2)
            if not any(orb_rect.colliderect(obs) for obs in self.obstacles):
                self.orbs.append({'pos': pos, 'type': orb_type, 'color': color})
                return

    def _get_observation(self):
        self._render_background()
        self._render_obstacles()
        self._render_orbs()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "speed_percent": int(self.speed_multiplier * 100),
            "green_orbs": self.green_orbs_collected,
            "red_orbs": self.red_orbs_collected,
            "level": self.level
        }
    
    # --- Rendering Methods ---

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            t = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - t) + self.COLOR_BG_BOTTOM[0] * t),
                int(self.COLOR_BG_TOP[1] * (1 - t) + self.COLOR_BG_BOTTOM[1] * t),
                int(self.COLOR_BG_TOP[2] * (1 - t) + self.COLOR_BG_BOTTOM[2] * t)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_obstacles(self):
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs, border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_OUTLINE, obs, width=2, border_radius=5)

    def _render_orbs(self):
        for orb in self.orbs:
            pos = (int(orb['pos'].x), int(orb['pos'].y))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ORB_RADIUS, orb['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ORB_RADIUS, orb['color'])

    def _render_player(self):
        pos = (int(self.player_pos.x), int(self.player_pos.y))
        
        # Flicker effect when invincible
        if self.invincibility_timer > 0 and (self.invincibility_timer // 3) % 2 == 0:
            return

        # Glow effect
        for i in range(self.PLAYER_RADIUS // 2, 0, -2):
            alpha = self.COLOR_PLAYER_GLOW[3] * (1 - i / (self.PLAYER_RADIUS // 2))
            glow_color = (*self.COLOR_PLAYER_GLOW[:3], int(alpha))
            s = pygame.Surface((self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, self.PLAYER_RADIUS, self.PLAYER_RADIUS, self.PLAYER_RADIUS + i, glow_color)
            self.screen.blit(s, (pos[0] - self.PLAYER_RADIUS, pos[1] - self.PLAYER_RADIUS))

        # Main ball
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _create_particle_burst(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'lifetime': self.np_random.integers(20, 40),
                'color': color,
                'size': self.np_random.uniform(1, 4)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / 40))
            color = (*p['color'][:3], alpha)
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
            self.screen.blit(s, (p['pos'].x - p['size'], p['pos'].y - p['size']))

    def _render_ui(self):
        # Score
        self._draw_text(f"SCORE: {self.score}", (10, 10))
        
        # Speed
        speed_text = f"SPEED: {int(self.speed_multiplier * 100)}%"
        speed_color = self.COLOR_GREEN_ORB if self.speed_multiplier >= 1.0 else self.COLOR_RED_ORB
        self._draw_text(speed_text, (self.SCREEN_WIDTH / 2 - self.font_ui.size(speed_text)[0]/2, 10), color=speed_color)
        
        # Lives
        for i in range(self.INITIAL_LIVES):
            pos_x = self.SCREEN_WIDTH - 30 - i * 25
            pos_y = 22
            if i < self.lives:
                color = self.COLOR_HEART
            else:
                color = self.COLOR_OBSTACLE
            
            points = [
                (pos_x, pos_y - 5), (pos_x + 7, pos_y - 12), (pos_x + 14, pos_y - 5),
                (pos_x + 7, pos_y + 5)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
    
    def _render_game_over_screen(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        message = "YOU WIN!" if self.win_condition_met else "GAME OVER"
        text_surf = self.font_game_over.render(message, True, self.COLOR_PLAYER)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
        self.screen.blit(text_surf, text_rect)

        final_score_text = f"Final Score: {self.score}"
        score_surf = self.font_ui.render(final_score_text, True, self.COLOR_UI_TEXT)
        score_rect = score_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 30))
        self.screen.blit(score_surf, score_rect)

    def _draw_text(self, text, pos, color=COLOR_UI_TEXT):
        shadow_surf = self.font_ui.render(text, True, self.COLOR_UI_SHADOW)
        self.screen.blit(shadow_surf, (pos[0] + 1, pos[1] + 1))
        text_surf = self.font_ui.render(text, True, color)
        self.screen.blit(text_surf, pos)
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

# Example usage:
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # To run manually, you need a different setup
    # This is just to show the env can be created and stepped through
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0
    
    env.close()

    # Manual play example:
    print("\nStarting manual play example...")
    try:
        env = GameEnv()
        obs, info = env.reset()
        pygame.display.set_caption("Bouncing Orb Collector")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        clock = pygame.time.Clock()
        
        running = True
        while running:
            movement = 0 # No-op
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                movement = 3
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                movement = 4

            action = [movement, 0, 0] # space/shift not used
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Display the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}")
                pygame.time.wait(2000) # Pause before reset
                obs, info = env.reset()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            clock.tick(60) # Run at 60 FPS for smooth visuals
    
    finally:
        env.close()