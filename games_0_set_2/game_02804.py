
# Generated: 2025-08-27T21:29:21.429650
# Source Brief: brief_02804.md
# Brief Index: 2804

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to select jump direction. Hold Shift for a short hop, or Space for a long jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop across procedurally generated platforms, dodging enemies, to reach the goal at the top of each level."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.WIDTH, self.HEIGHT = 640, 400

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
        self.font_ui = pygame.font.SysFont("sans-serif", 24)
        self.font_big = pygame.font.SysFont("sans-serif", 48)

        # Colors
        self.COLOR_BG_TOP = (40, 20, 80)
        self.COLOR_BG_BOTTOM = (80, 60, 150)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_SHADOW = (0, 0, 0, 100)
        self.COLOR_PLATFORM = (60, 120, 255)
        self.COLOR_GOAL = (255, 215, 0)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PARTICLE_JUMP = (200, 200, 200)
        self.COLOR_PARTICLE_LAND = (255, 215, 0)
        self.COLOR_PARTICLE_DEATH = self.COLOR_ENEMY

        # Game constants
        self.GRAVITY = 0.4
        self.MAX_FALL_SPEED = 8
        self.MAX_TIME = int(60 * 30) # 60 seconds at 30 FPS

        # Pre-render background for performance
        self.background_surf = self._create_gradient_background()

        # Initialize state variables
        self.reset()
        
        # self.validate_implementation() # Call for self-check

    def _create_gradient_background(self):
        """Creates a surface with a vertical gradient."""
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = [
                int(self.COLOR_BG_TOP[i] * (1 - ratio) + self.COLOR_BG_BOTTOM[i] * ratio)
                for i in range(3)
            ]
            pygame.draw.line(bg, color, (0, y), (self.WIDTH, y))
        return bg

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
            random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 3
        self.stage = 1
        self.timer = self.MAX_TIME
        self.camera_y = 0

        self.player = {
            'rect': pygame.Rect(0, 0, 20, 20),
            'vel': [0, 0],
            'on_platform_idx': 0,
            'is_airborne': True
        }

        self.platforms = []
        self.enemies = []
        self.particles = []

        self._setup_stage()
        self.camera_y = self.player['rect'].centery - self.HEIGHT * 0.75

        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        """Generates platforms, enemies, and sets player position for the current stage."""
        self.platforms.clear()
        self.enemies.clear()
        self.particles.clear()
        self.timer = self.MAX_TIME

        # Generate platforms
        start_plat = pygame.Rect(self.WIDTH // 2 - 50, self.HEIGHT - 40, 100, 20)
        self.platforms.append({'rect': start_plat, 'type': 'normal'})
        last_plat = start_plat
        for i in range(1, 20):
            w = random.randint(60, 120)
            h = 20
            px, py = last_plat.center
            
            dx = random.randint(-150, 150)
            dy = random.randint(80, 130)

            x = px + dx - w / 2
            y = py - dy - h / 2
            
            # Clamp to screen bounds
            x = max(20, min(x, self.WIDTH - w - 20))

            new_plat_rect = pygame.Rect(x, y, w, h)
            
            # Ensure no overlap
            is_overlapping = any(new_plat_rect.colliderect(p['rect'].inflate(20, 20)) for p in self.platforms)
            if not is_overlapping:
                self.platforms.append({'rect': new_plat_rect, 'type': 'normal'})
                last_plat = new_plat_rect
        
        self.platforms[-1]['type'] = 'goal'

        # Place enemies
        enemy_speed = 1.0 + (self.stage - 1) * 0.2
        for i, p_data in enumerate(self.platforms):
            if i > 0 and i < len(self.platforms) -1 and random.random() < 0.4:
                plat_rect = p_data['rect']
                self.enemies.append({
                    'rect': pygame.Rect(plat_rect.x, plat_rect.y - 15, 15, 15),
                    'vel_x': enemy_speed if random.random() > 0.5 else -enemy_speed,
                    'bounds': (plat_rect.left, plat_rect.right - 15)
                })

        self._reset_player_to_start()

    def _reset_player_to_start(self):
        """Resets player to the start of the current stage."""
        start_plat_rect = self.platforms[0]['rect']
        self.player['rect'].midbottom = start_plat_rect.midtop
        self.player['vel'] = [0, 0]
        self.player['on_platform_idx'] = 0
        self.player['is_airborne'] = False
        self.camera_y += (self.player['rect'].centery - self.HEIGHT * 0.75 - self.camera_y) * 0.5

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.1  # Survival reward
        
        self._update_player(action)
        self._update_enemies()
        self._update_particles()
        
        stage_cleared, terminal_win = self._handle_collisions()
        reward += stage_cleared * 10
        if terminal_win:
            reward += 100
            
        # Handle falling or death
        lost_life = False
        if self.player['rect'].top > self.camera_y + self.HEIGHT: # Fallen off bottom
            reward -= 10
            lost_life = True
        
        # Check enemy collisions
        for enemy in self.enemies:
            if self.player['rect'].colliderect(enemy['rect']):
                reward -= 10
                lost_life = True
                self._create_particles(self.player['rect'].center, 20, self.COLOR_PARTICLE_DEATH, 3, 5)
                # SFX: Player hit
                break

        if lost_life:
            self.lives -= 1
            if self.lives > 0:
                self._reset_player_to_start()
            else:
                self.game_over = True
        
        # Timer
        self.timer -= 1
        if self.timer <= 0:
            self.lives -= 1
            if self.lives > 0:
                self.timer = self.MAX_TIME
                self._reset_player_to_start()
            else:
                self.game_over = True

        self.steps += 1
        self.score += reward
        terminated = self.game_over or self.lives <= 0
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Handle jumping input only if on a platform
        if not self.player['is_airborne'] and movement != 0:
            self.player['is_airborne'] = True
            
            jump_power = 9.0
            move_speed = 4.0

            if space_held: # Long jump
                jump_power = 12.0
                move_speed = 5.0
            elif shift_held: # Short hop
                jump_power = 6.0
                move_speed = 3.0

            jump_dir = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement, (0,0))
            self.player['vel'][0] = jump_dir[0] * move_speed
            self.player['vel'][1] = jump_dir[1] * jump_power if jump_dir[1] != 0 else -jump_power
            
            # SFX: Jump
            self._create_particles(self.player['rect'].midbottom, 10, self.COLOR_PARTICLE_JUMP, 1, 3)

        # Apply physics
        if self.player['is_airborne']:
            self.player['vel'][1] += self.GRAVITY
            self.player['vel'][1] = min(self.player['vel'][1], self.MAX_FALL_SPEED)
        
        self.player['rect'].x += self.player['vel'][0]
        self.player['rect'].y += self.player['vel'][1]

    def _handle_collisions(self):
        """Checks and resolves player collisions with platforms."""
        self.player['is_airborne'] = True
        stage_cleared = False
        terminal_win = False

        for i, p_data in enumerate(self.platforms):
            plat_rect = p_data['rect']
            if self.player['rect'].colliderect(plat_rect) and self.player['vel'][1] >= 0:
                # Check if player was above the platform in the previous frame
                if self.player['rect'].bottom - self.player['vel'][1] <= plat_rect.top:
                    self.player['rect'].bottom = plat_rect.top
                    self.player['vel'] = [0, 0]
                    self.player['is_airborne'] = False
                    
                    if i != self.player['on_platform_idx']:
                        # SFX: Land
                        self._create_particles(self.player['rect'].midbottom, 15, self.COLOR_PARTICLE_LAND, 2, 4)
                        if i == self.player['on_platform_idx'] + 1:
                            self.score -= 0.2 # Penalty for safe jump
                    
                    self.player['on_platform_idx'] = i

                    if p_data['type'] == 'goal':
                        stage_cleared = True
                        # SFX: Stage Clear
                        if self.stage < 3:
                            self.stage += 1
                            self._setup_stage()
                        else:
                            self.game_over = True
                            terminal_win = True
                    break
        return stage_cleared, terminal_win

    def _update_enemies(self):
        for enemy in self.enemies:
            enemy['rect'].x += enemy['vel_x']
            if enemy['rect'].left <= enemy['bounds'][0] or enemy['rect'].right >= enemy['bounds'][1]:
                enemy['vel_x'] *= -1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _create_particles(self, pos, count, color, speed_min, speed_max):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(speed_min, speed_max)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(10, 20),
                'color': color
            })

    def _get_observation(self):
        # Update camera
        target_cam_y = self.player['rect'].centery - self.HEIGHT * 0.7
        self.camera_y += (target_cam_y - self.camera_y) * 0.1 # Smooth follow

        # Render all game elements
        self.screen.blit(self.background_surf, (0, 0))
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
            "timer": self.timer / 30.0
        }

    def _render_game(self):
        # Render platforms
        for p_data in self.platforms:
            color = self.COLOR_GOAL if p_data['type'] == 'goal' else self.COLOR_PLATFORM
            cam_rect = p_data['rect'].move(0, -self.camera_y)
            pygame.draw.rect(self.screen, color, cam_rect, border_radius=3)

        # Render enemies
        for enemy in self.enemies:
            cam_rect = enemy['rect'].move(0, -self.camera_y)
            pygame.gfxdraw.aacircle(self.screen, int(cam_rect.centerx), int(cam_rect.centery), int(cam_rect.width/2), self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, int(cam_rect.centerx), int(cam_rect.centery), int(cam_rect.width/2), self.COLOR_ENEMY)

        # Render player shadow
        if self.player['is_airborne']:
            shadow_rect = self.player['rect'].copy()
            shadow_rect.height = 5
            shadow_rect.midtop = self.player['rect'].midbottom
            shadow_surf = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surf, self.COLOR_SHADOW, (0, 0, shadow_rect.width, shadow_rect.height))
            self.screen.blit(shadow_surf, shadow_rect.move(0, -self.camera_y))

        # Render player
        cam_rect = self.player['rect'].move(0, -self.camera_y)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, cam_rect, border_radius=3)

        # Render particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1] - self.camera_y))
            alpha = int(255 * (p['life'] / 20.0))
            color = p['color'] + (alpha,)
            pygame.draw.circle(self.screen, color, pos, int(p['life'] / 5))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, self.timer / 30.0)
        time_color = self.COLOR_ENEMY if time_left < 10 else self.COLOR_TEXT
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, time_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Lives
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (10, self.HEIGHT - lives_text.get_height() - 10))

        # Game Over / Win Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.lives > 0 else "GAME OVER"
            win_text = self.font_big.render(msg, True, self.COLOR_GOAL if self.lives > 0 else self.COLOR_ENEMY)
            text_rect = win_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(win_text, text_rect)

    def render(self):
        if self.auto_advance:
            self.clock.tick(30)
        return self._get_observation()
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# Example usage to run and display the game
if __name__ == '__main__':
    # Set this to False to use keyboard controls
    USE_RANDOM_AGENT = True

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a pygame window for display
    pygame.display.set_caption("Hopper Game")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    total_reward = 0
    
    # Action state
    action = [0, 0, 0] # no-op, no-space, no-shift

    while running:
        if not USE_RANDOM_AGENT:
            # Keyboard control
            action = [0, 0, 0]
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        else:
            # Random agent
            action = env.action_space.sample()

        # Handle window closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print("Resetting environment...")
            total_reward = 0
            obs, info = env.reset()
            # Give a moment to see the game over screen
            pygame.time.wait(2000)

    env.close()