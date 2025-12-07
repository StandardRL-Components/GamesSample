import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to fly the bird. Dodge the trees and collect the glowing orbs."
    )

    game_description = (
        "Navigate a bird through a procedurally generated forest. Dodge obstacles, collect rewards, "
        "and try to survive through three increasingly fast stages."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    STAGE_LENGTH = 100  # "meters"
    MAX_STEPS = 10000

    # Colors
    COLOR_BG = (20, 40, 60)
    COLOR_BG_PARALLAX = (30, 60, 80)
    COLOR_OBSTACLE = (80, 50, 30)
    COLOR_OBSTACLE_OUTLINE = (50, 30, 20)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_OUTLINE = (200, 255, 255)
    COLOR_REWARD = (255, 220, 0)
    COLOR_REWARD_OUTLINE = (255, 255, 150)
    COLOR_TEXT = (255, 255, 255)
    COLOR_PARTICLE_CRASH = (255, 50, 50)
    COLOR_PARTICLE_REWARD = (255, 240, 100)

    # Physics
    PLAYER_ACCEL = 0.8
    PLAYER_DAMPING = 0.85
    PLAYER_MAX_SPEED = 8.0
    PLAYER_RADIUS = 10

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_gameover = pygame.font.Font(None, 72)

        self.obstacles = []
        self.rewards = []
        self.particles = []
        self.parallax_objects = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.stage = 1
        self.distance_in_stage = 0.0
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT * 0.8], dtype=np.float64)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float64)

        self.obstacles.clear()
        self.rewards.clear()
        self.particles.clear()
        self.parallax_objects.clear()
        
        self.last_gap_center = self.WIDTH / 2
        
        for i in range(10):
            self._spawn_parallax(y_pos=self.np_random.integers(0, self.HEIGHT))
        
        # FIX: Spawn obstacles ahead of the player, not on top of them.
        for i in range(5):
            self._spawn_obstacle(y_pos=100 - i * 150)
        
        self._spawn_reward()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If already game over, just return the final state
            truncated = self.steps >= self.MAX_STEPS
            return self._get_observation(), 0, True, truncated, self._get_info()

        self.steps += 1
        
        # --- Update Player ---
        self._update_player(action)

        # --- Update World ---
        scroll_speed = 1.0 + (self.stage - 1) * 0.2
        scroll_speed *= 4 # Scale to feel better
        
        self.distance_in_stage += scroll_speed / self.FPS

        event_reward = self._update_world_objects(scroll_speed)

        # --- Spawning ---
        self._manage_spawning()
        
        # --- Progression and Rewards ---
        reward = 0.1  # Survival reward
        reward += event_reward
        
        progression_reward, stage_cleared = self._manage_progression()
        reward += progression_reward
        
        if stage_cleared:
            # SFX: Stage clear fanfare
            pass

        # --- Termination ---
        # FIX: Separate termination (game end) from truncation (time limit)
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS

        if self.game_over and not self.win:
            # SFX: Player crash/explosion
            pass
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _update_player(self, action):
        movement = action[0]

        if movement == 1:  # Up
            self.player_vel[1] -= self.PLAYER_ACCEL
        elif movement == 2:  # Down
            self.player_vel[1] += self.PLAYER_ACCEL
        if movement == 3:  # Left
            self.player_vel[0] -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vel[0] += self.PLAYER_ACCEL

        self.player_vel = np.clip(self.player_vel, -self.PLAYER_MAX_SPEED, self.PLAYER_MAX_SPEED)
        self.player_pos += self.player_vel
        self.player_vel *= self.PLAYER_DAMPING

        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)

    def _update_world_objects(self, scroll_speed):
        reward = 0

        # Update obstacles
        for obs in self.obstacles:
            obs['rect'].y += scroll_speed
            if obs['rect'].colliderect(self._get_player_rect()):
                self.game_over = True
                self._create_particles(self.player_pos, 30, self.COLOR_PARTICLE_CRASH, 8, 20)
            
            # Near miss check
            close_rect = obs['rect'].inflate(10, 10)
            if close_rect.colliderect(self._get_player_rect()) and not obs['near_miss_triggered']:
                reward -= 1.0
                obs['near_miss_triggered'] = True
        
        self.obstacles = [obs for obs in self.obstacles if obs['rect'].top < self.HEIGHT]

        # Update rewards
        for rwd in self.rewards:
            rwd['pos'][1] += scroll_speed
            if self._check_circle_collision(self.player_pos, self.PLAYER_RADIUS, rwd['pos'], rwd['radius']):
                reward += 5.0
                self.score += 500
                rwd['collected'] = True
                self._create_particles(rwd['pos'], 20, self.COLOR_PARTICLE_REWARD, 5, 15)
                # SFX: Reward collected
        
        self.rewards = [rwd for rwd in self.rewards if rwd['pos'][1] < self.HEIGHT + rwd['radius'] and not rwd.get('collected')]
        
        # Update particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

        # Update parallax
        for p_obj in self.parallax_objects:
            p_obj['pos'][1] += scroll_speed * 0.5
        self.parallax_objects = [p for p in self.parallax_objects if p['pos'][1] < self.HEIGHT + p['radius']]
        
        return reward

    def _manage_spawning(self):
        if not self.obstacles or self.obstacles[-1]['rect'].top > 100:
            self._spawn_obstacle()
        
        if len(self.rewards) < 2 and self.np_random.random() < 0.02:
            self._spawn_reward()
            
        if len(self.parallax_objects) < 10 and self.np_random.random() < 0.1:
            self._spawn_parallax()

    def _manage_progression(self):
        reward = 0
        stage_cleared = False
        if self.distance_in_stage >= self.STAGE_LENGTH:
            self.stage += 1
            self.distance_in_stage = 0
            stage_cleared = True
            if self.stage > 3:
                self.win = True
                self.game_over = True
                reward += 500  # Win bonus
                self.score += 5000
            else:
                reward += 100  # Stage clear bonus
                self.score += 1000
        return reward, stage_cleared

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_obstacles()
        self._render_rewards()
        self._render_particles()
        if not (self.game_over and not self.win):
             self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for p_obj in self.parallax_objects:
            pygame.gfxdraw.filled_circle(
                self.screen, int(p_obj['pos'][0]), int(p_obj['pos'][1]), int(p_obj['radius']), self.COLOR_BG_PARALLAX
            )

    def _render_obstacles(self):
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'])
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_OUTLINE, obs['rect'], 2)

    def _render_rewards(self):
        for rwd in self.rewards:
            pos = (int(rwd['pos'][0]), int(rwd['pos'][1]))
            radius = int(rwd['radius'])
            anim_scale = 0.8 + 0.2 * math.sin(self.steps * 0.2)
            
            # Pulsing glow
            glow_radius = int(radius * 1.5 * anim_scale)
            glow_color = (*self.COLOR_REWARD, 60) # RGBA
            glow_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, glow_color, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (pos[0]-glow_radius, pos[1]-glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_REWARD)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_REWARD_OUTLINE)

    def _render_player(self):
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        
        # Flapping wings animation
        wing_angle = math.sin(self.steps * 0.5) * 0.6
        wing_len = self.PLAYER_RADIUS * 1.5
        
        # Left wing
        lx1, ly1 = pos[0] - self.PLAYER_RADIUS * 0.5, pos[1]
        lx2, ly2 = lx1 - wing_len * math.cos(wing_angle), ly1 - wing_len * math.sin(wing_angle)
        pygame.draw.line(self.screen, self.COLOR_PLAYER_OUTLINE, (lx1, ly1), (lx2, ly2), 4)
        
        # Right wing
        rx1, ry1 = pos[0] + self.PLAYER_RADIUS * 0.5, pos[1]
        rx2, ry2 = rx1 + wing_len * math.cos(wing_angle), ry1 - wing_len * math.sin(wing_angle)
        pygame.draw.line(self.screen, self.COLOR_PLAYER_OUTLINE, (rx1, ry1), (rx2, ry2), 4)

        # Body with outline
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER_OUTLINE)

    def _render_particles(self):
        for p in self.particles:
            size = max(1, int(p['lifetime'] / p['max_lifetime'] * p['size']))
            pygame.draw.circle(self.screen, p['color'], p['pos'], size)

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        stage_text = self.font_small.render(f"Stage: {self.stage}/3", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, 45))
        
        progress = self.distance_in_stage / self.STAGE_LENGTH
        pygame.draw.rect(self.screen, (100, 100, 100), (10, 70, 150, 10))
        pygame.draw.rect(self.screen, self.COLOR_REWARD, (10, 70, 150 * progress, 10))

        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_gameover.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "distance_in_stage": self.distance_in_stage,
        }

    def _get_player_rect(self):
        return pygame.Rect(
            self.player_pos[0] - self.PLAYER_RADIUS,
            self.player_pos[1] - self.PLAYER_RADIUS,
            self.PLAYER_RADIUS * 2,
            self.PLAYER_RADIUS * 2
        )

    def _check_circle_collision(self, pos1, r1, pos2, r2):
        return np.linalg.norm(pos1 - pos2) < (r1 + r2)

    def _spawn_obstacle(self, y_pos=-100):
        gap_width = self.PLAYER_RADIUS * 7
        min_gap_center = gap_width / 2 + 20
        max_gap_center = self.WIDTH - gap_width / 2 - 20
        
        # Make gap wander smoothly
        target_gap_center = self.np_random.uniform(min_gap_center, max_gap_center)
        self.last_gap_center = self.last_gap_center * 0.8 + target_gap_center * 0.2

        gap_center_x = self.last_gap_center
        gap_start = gap_center_x - gap_width / 2
        gap_end = gap_center_x + gap_width / 2
        
        height = 80

        # Left obstacle
        if gap_start > 0:
            self.obstacles.append({
                'rect': pygame.Rect(0, y_pos, gap_start, height),
                'near_miss_triggered': False
            })
        
        # Right obstacle
        if gap_end < self.WIDTH:
            self.obstacles.append({
                'rect': pygame.Rect(gap_end, y_pos, self.WIDTH - gap_end, height),
                'near_miss_triggered': False
            })

    def _spawn_reward(self):
        if not self.obstacles: return
        
        # Spawn in a safe horizontal zone
        x = self.np_random.integers(50, self.WIDTH - 50)
        y = -50
        
        self.rewards.append({
            'pos': np.array([x, y], dtype=np.float64),
            'radius': 8,
        })
        
    def _spawn_parallax(self, y_pos=None):
        if y_pos is None:
            y_pos = -50
        self.parallax_objects.append({
            'pos': np.array([self.np_random.uniform(0, self.WIDTH), y_pos], dtype=np.float64),
            'radius': self.np_random.uniform(20, 80)
        })

    def _create_particles(self, pos, count, color, speed_max, lifetime_max):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_max)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifetime = self.np_random.integers(10, lifetime_max)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'color': color,
                'lifetime': lifetime,
                'max_lifetime': lifetime,
                'size': self.np_random.uniform(2, 5)
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not be run by the test suite
    
    # Unset the dummy video driver to allow for display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Forest Flyer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        # Convert to MultiDiscrete action
        action = [movement, 0, 0] # Space and Shift are not used

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for 'R' to restart
            pass
        
        clock.tick(GameEnv.FPS)

    env.close()