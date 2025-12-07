
# Generated: 2025-08-28T02:46:02.412344
# Source Brief: brief_01807.md
# Brief Index: 1807

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Stomp on bugs to score points."
    )

    game_description = (
        "A retro arcade platformer. Squash 20 bugs by jumping on them, but don't let them touch your sides! You have 3 lives."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.PLATFORM_H = 40
        self.GRAVITY = 0.6
        self.PLAYER_JUMP_STRENGTH = -11
        self.PLAYER_MOVE_SPEED = 5
        self.PLAYER_W, self.PLAYER_H = 24, 32
        self.BUG_W, self.BUG_H = 20, 10
        self.MAX_BUGS = 6
        self.WIN_SCORE = 20
        self.MAX_LIVES = 3
        self.MAX_STEPS = 1500 # 50 seconds at 30fps
        self.INVINCIBILITY_FRAMES = 60 # 2 seconds

        # Colors
        self.COLOR_BG_SKY = (50, 120, 200)
        self.COLOR_HILL_1 = (30, 80, 130)
        self.COLOR_HILL_2 = (40, 100, 160)
        self.COLOR_PLATFORM = (100, 60, 20)
        self.COLOR_PLATFORM_TOP = (120, 180, 50)
        self.COLOR_PLAYER = (100, 255, 100)
        self.COLOR_BUG = (255, 50, 50)
        self.COLOR_SPLAT = (200, 0, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_SHADOW = (0, 0, 0, 100)
        self.COLOR_HEART = (255, 80, 80)

        # Fonts
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        self.bugs = None
        self.splats = None
        self.bug_speed = None
        self.steps = None
        self.score = None
        self.bugs_squashed = None
        self.lives = None
        self.game_over = None
        self.win = None
        self.invincibility_timer = None
        self.terminal_reward_given = None
        self.background_hills = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.bugs_squashed = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.win = False
        self.terminal_reward_given = False
        self.invincibility_timer = 0

        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - self.PLATFORM_H - self.PLAYER_H)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = True
        
        self.bug_speed = 1.0
        self.bugs = []
        for _ in range(self.MAX_BUGS):
            self._spawn_bug()

        self.splats = []

        if self.background_hills is None:
            self.background_hills = []
            for _ in range(10):
                self.background_hills.append({
                    "pos": (random.randint(0, self.WIDTH), random.randint(self.HEIGHT // 2, self.HEIGHT - self.PLATFORM_H)),
                    "radius": random.randint(50, 150),
                    "color": random.choice([self.COLOR_HILL_1, self.COLOR_HILL_2])
                })

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If game is over, do nothing but return the final state
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()
        
        reward = -0.01  # Small penalty per step to encourage efficiency

        # 1. Handle player input
        movement = action[0]
        self._handle_input(movement)
        
        # 2. Update game state
        self._update_player()
        self._update_bugs()
        reward += self._handle_collisions()
        self._update_effects()

        # 3. Check for termination conditions
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.terminal_reward_given:
            if self.win:
                reward += 100
            else:
                reward -= 100
            self.terminal_reward_given = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.player_vel.x = -self.PLAYER_MOVE_SPEED
        elif movement == 4:  # Right
            self.player_vel.x = self.PLAYER_MOVE_SPEED
        else:
            self.player_vel.x = 0
        
        if movement == 1 and self.on_ground:  # Jump
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH
            self.on_ground = False
            # Sound: Player jump

    def _update_player(self):
        # Apply gravity
        self.player_vel.y += self.GRAVITY
        
        # Update position
        self.player_pos += self.player_vel
        
        # Ground collision
        platform_y = self.HEIGHT - self.PLATFORM_H
        if self.player_pos.y + self.PLAYER_H > platform_y:
            self.player_pos.y = platform_y - self.PLAYER_H
            self.player_vel.y = 0
            self.on_ground = True
            
        # Screen boundaries
        self.player_pos.x = max(0, min(self.WIDTH - self.PLAYER_W, self.player_pos.x))

        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1
    
    def _update_bugs(self):
        for bug in self.bugs:
            bug['pos'].x += bug['vel_x']
            # Simple crawling animation state
            bug['anim_state'] = (bug['anim_state'] + 0.25) % 2

            # Screen wrapping
            if bug['vel_x'] > 0 and bug['pos'].x > self.WIDTH:
                bug['pos'].x = -self.BUG_W
            elif bug['vel_x'] < 0 and bug['pos'].x < -self.BUG_W:
                bug['pos'].x = self.WIDTH

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_W, self.PLAYER_H)
        
        bugs_to_remove = []
        for bug in self.bugs:
            bug_rect = pygame.Rect(bug['pos'].x, bug['pos'].y, self.BUG_W, self.BUG_H)
            if player_rect.colliderect(bug_rect):
                # Check for squash (player falling and is above the bug)
                is_squash = self.player_vel.y > 0 and (player_rect.bottom < bug_rect.centery)
                
                if is_squash:
                    # Sound: Bug squash
                    reward += 10
                    self.score += 10
                    self.bugs_squashed += 1
                    
                    # Add a splat effect
                    self.splats.append({'pos': bug['pos'].copy(), 'timer': 30})
                    bugs_to_remove.append(bug)
                    
                    # Small bounce
                    self.player_vel.y = self.PLAYER_JUMP_STRENGTH * 0.5
                    
                    # Increase difficulty
                    if self.bugs_squashed > 0 and self.bugs_squashed % 5 == 0:
                        self.bug_speed = min(3.0, self.bug_speed + 0.1)

                elif self.invincibility_timer == 0:
                    # Side collision, player gets hurt
                    # Sound: Player hurt
                    reward -= 10
                    self.lives -= 1
                    self.invincibility_timer = self.INVINCIBILITY_FRAMES
                    # Knockback effect on player
                    self.player_vel.y = self.PLAYER_JUMP_STRENGTH * 0.4
                    self.player_vel.x = -self.player_vel.x if self.player_vel.x != 0 else (1 if self.player_pos.x < self.WIDTH / 2 else -1) * self.PLAYER_MOVE_SPEED * 0.5

        if bugs_to_remove:
            self.bugs = [b for b in self.bugs if b not in bugs_to_remove]
            for _ in bugs_to_remove:
                self._spawn_bug()
        
        return reward

    def _update_effects(self):
        # Update splats
        self.splats = [s for s in self.splats if s['timer'] > 0]
        for s in self.splats:
            s['timer'] -= 1

    def _spawn_bug(self):
        side = random.choice([-1, 1])
        start_x = -self.BUG_W if side == 1 else self.WIDTH
        pos = pygame.Vector2(start_x, self.HEIGHT - self.PLATFORM_H - self.BUG_H)
        vel_x = self.bug_speed * side * random.uniform(0.8, 1.2)
        self.bugs.append({'pos': pos, 'vel_x': vel_x, 'anim_state': 0})
        
    def _check_termination(self):
        if self.bugs_squashed >= self.WIN_SCORE:
            self.game_over = True
            self.win = True
            return True
        if self.lives <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False
        
    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG_SKY)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background hills
        for hill in self.background_hills:
            pygame.gfxdraw.filled_circle(self.screen, int(hill["pos"][0]), int(hill["pos"][1]), hill["radius"], hill["color"])

        # Platform
        platform_rect = pygame.Rect(0, self.HEIGHT - self.PLATFORM_H, self.WIDTH, self.PLATFORM_H)
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, platform_rect)
        pygame.draw.line(self.screen, self.COLOR_PLATFORM_TOP, (0, self.HEIGHT - self.PLATFORM_H), (self.WIDTH, self.HEIGHT - self.PLATFORM_H), 3)

        # Splats
        for s in self.splats:
            radius = int(self.BUG_W * 0.75 * (s['timer'] / 30.0))
            if radius > 1:
                pygame.gfxdraw.filled_circle(self.screen, int(s['pos'].x + self.BUG_W / 2), int(s['pos'].y + self.BUG_H), radius, self.COLOR_SPLAT)

        # Bugs
        for bug in self.bugs:
            bug_rect = pygame.Rect(int(bug['pos'].x), int(bug['pos'].y), self.BUG_W, self.BUG_H)
            pygame.draw.ellipse(self.screen, self.COLOR_BUG, bug_rect)
            # Legs animation
            leg_y = bug_rect.bottom
            for i in range(3):
                offset = (math.sin(self.steps * 0.2 + i) * 3) if int(bug['anim_state']) == 0 else (math.cos(self.steps * 0.2 + i) * 3)
                start_pos = (bug_rect.centerx + (i - 1) * 5 - 3, leg_y - 2)
                end_pos = (start_pos[0] + offset, leg_y + 3)
                pygame.draw.line(self.screen, self.COLOR_BUG, start_pos, end_pos, 2)

        # Player
        if self.invincibility_timer == 0 or self.steps % 6 < 3:
            player_rect = pygame.Rect(int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_W, self.PLAYER_H)
            
            # Stretch and squash animation
            if not self.on_ground:
                squash_factor = self.player_vel.y * 0.3
                player_rect.height = max(10, self.PLAYER_H - squash_factor)
                player_rect.width = max(10, self.PLAYER_W + squash_factor)
                player_rect.x = int(self.player_pos.x - (player_rect.width - self.PLAYER_W) / 2)
                player_rect.y = int(self.player_pos.y + (self.PLAYER_H - player_rect.height))

            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
            # Eyes
            eye_y = player_rect.y + 8
            eye_x_offset = 5
            eye_dir = 1 if self.player_vel.x >= 0 else -1
            pygame.draw.circle(self.screen, (255,255,255), (player_rect.centerx - eye_x_offset, eye_y), 3)
            pygame.draw.circle(self.screen, (255,255,255), (player_rect.centerx + eye_x_offset, eye_y), 3)
            pygame.draw.circle(self.screen, (0,0,0), (player_rect.centerx - eye_x_offset + eye_dir, eye_y), 1)
            pygame.draw.circle(self.screen, (0,0,0), (player_rect.centerx + eye_x_offset + eye_dir, eye_y), 1)

    def _render_ui(self):
        # Score
        score_surf = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))
        
        # Lives
        for i in range(self.lives):
            self._draw_heart(self.WIDTH - 30 - i * 35, 25, 12)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_SHADOW)
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            msg_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _draw_heart(self, x, y, size):
        points = [
            (x, y - size // 4),
            (x - size // 2, y - size // 2),
            (x - size // 2, y),
            (x, y + size // 2),
            (x + size // 2, y),
            (x + size // 2, y - size // 2)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_HEART)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_HEART)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bugs_squashed": self.bugs_squashed,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Bug Squasher")
    
    done = False
    clock = pygame.time.Clock()
    total_reward = 0
    
    while not done:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]: # No effect, but maps to an action
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_SHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
    env.close()