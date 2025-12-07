
# Generated: 2025-08-28T05:47:13.256149
# Source Brief: brief_05692.md
# Brief Index: 5692

        
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
        "Controls: Use arrow keys (↑↓←→) to pilot the submersible. "
        "Collect all the green trash while avoiding the red sea mines."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a submersible to collect underwater trash for a high score. "
        "Each piece of trash increases your score, but colliding with a sea mine is dangerous. "
        "The mission ends if you hit 5 mines or collect all 50 pieces of trash."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    WORLD_WIDTH, WORLD_HEIGHT = 1280, 800
    FPS = 30
    MAX_STEPS = 1500

    # Colors
    COLOR_BG_TOP = (20, 40, 80)
    COLOR_BG_BOTTOM = (10, 20, 40)
    COLOR_SUB = (255, 200, 0)
    COLOR_SUB_WINDOW = (150, 220, 255)
    COLOR_SUB_GLOW = (255, 200, 0, 50)
    COLOR_TRASH = (50, 255, 150)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_OBSTACLE_SPIKE = (200, 20, 20)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0)

    # Game parameters
    NUM_TRASH = 50
    NUM_OBSTACLES = 15
    MAX_HITS = 5
    PLAYER_SPEED = 5
    PLAYER_DRAG = 0.92
    PLAYER_RADIUS = 15
    TRASH_RADIUS = 8
    OBSTACLE_RADIUS = 12

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
        self.font_main = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 18, bold=True)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.trash = None
        self.obstacles = None
        self.particles = None
        self.bubbles = None
        self.camera_pos = None
        self.steps = 0
        self.score = 0
        self.hits = 0
        self.game_over = False
        self.last_dist_to_trash = 0
        self.hit_cooldown = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.player_pos = np.array([self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)

        self.trash = self._generate_entities(self.NUM_TRASH, self.TRASH_RADIUS * 2)
        self.obstacles = self._generate_obstacles(self.NUM_OBSTACLES, self.OBSTACLE_RADIUS * 4)
        
        self.particles = []
        self.bubbles = [self._create_bubble() for _ in range(30)]
        self.camera_pos = self.player_pos - np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2])
        
        self.steps = 0
        self.score = 0
        self.hits = 0
        self.game_over = False
        self.hit_cooldown = 0

        self.last_dist_to_trash = self._get_dist_to_nearest_trash()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        
        step_reward = 0
        dist_before = self._get_dist_to_nearest_trash()

        self._update_player(movement)
        self._update_entities()
        
        collision_reward = self._handle_collisions()
        step_reward += collision_reward

        dist_after = self._get_dist_to_nearest_trash()

        if dist_before > 0:
            if dist_after < dist_before:
                step_reward += 1.0  # Moving closer
            else:
                step_reward -= 0.1  # Moving away

        self.last_dist_to_trash = dist_after
        
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.hits >= self.MAX_HITS or self.steps >= self.MAX_STEPS:
                step_reward -= 100  # Loss penalty
            elif not self.trash:
                step_reward += 100  # Win bonus

        return self._get_observation(), step_reward, terminated, False, self._get_info()

    def _generate_entities(self, count, min_dist):
        entities = []
        player_start_area = self.PLAYER_RADIUS * 5
        for _ in range(count):
            while True:
                pos = self.np_random.uniform(
                    low=[min_dist, min_dist],
                    high=[self.WORLD_WIDTH - min_dist, self.WORLD_HEIGHT - min_dist]
                )
                # Ensure not too close to player start
                if np.linalg.norm(pos - np.array([self.WORLD_WIDTH/2, self.WORLD_HEIGHT/2])) < player_start_area:
                    continue
                # Ensure not too close to other entities
                if all(np.linalg.norm(pos - e['pos']) > min_dist for e in entities):
                    entities.append({'pos': pos, 'anim_offset': self.np_random.uniform(0, 2 * math.pi)})
                    break
        return entities

    def _generate_obstacles(self, count, min_dist):
        obstacles = self._generate_entities(count, min_dist)
        for obs in obstacles:
            obs['base_pos'] = obs['pos'].copy()
            obs['amplitude'] = self.np_random.uniform(30, 80)
            obs['frequency'] = self.np_random.uniform(0.01, 0.03)
            obs['phase'] = self.np_random.uniform(0, 2 * math.pi)
            obs['axis'] = self.np_random.choice([0, 1]) # 0 for x-axis, 1 for y-axis
        return obstacles

    def _update_player(self, movement):
        # 1=up, 2=down, 3=left, 4=right
        if movement == 1: self.player_vel[1] -= 1
        elif movement == 2: self.player_vel[1] += 1
        elif movement == 3: self.player_vel[0] -= 1
        elif movement == 4: self.player_vel[0] += 1
        
        self.player_vel = np.clip(self.player_vel, -self.PLAYER_SPEED, self.PLAYER_SPEED)
        self.player_pos += self.player_vel
        self.player_vel *= self.PLAYER_DRAG

        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.WORLD_WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.WORLD_HEIGHT - self.PLAYER_RADIUS)

    def _update_entities(self):
        # Update obstacles (sinusoidal movement)
        for obs in self.obstacles:
            offset = obs['amplitude'] * math.sin(self.steps * obs['frequency'] + obs['phase'])
            obs['pos'] = obs['base_pos'].copy()
            obs['pos'][obs['axis']] += offset

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        # Update bubbles
        for b in self.bubbles:
            b['pos'][1] -= b['speed']
            b['pos'][0] += math.sin(b['pos'][1] * b['sway'])
            if b['pos'][1] < 0:
                b.update(self._create_bubble(at_bottom=True))
        
        if self.hit_cooldown > 0:
            self.hit_cooldown -= 1

    def _handle_collisions(self):
        reward = 0
        
        # Player vs Trash
        collected_indices = []
        for i, item in enumerate(self.trash):
            dist = np.linalg.norm(self.player_pos - item['pos'])
            if dist < self.PLAYER_RADIUS + self.TRASH_RADIUS:
                collected_indices.append(i)
                self.score += 10
                reward += 10
                # sfx: collect_sound
                self._create_effect(item['pos'], self.COLOR_TRASH, 20)
        
        # Remove collected trash in reverse order
        for i in sorted(collected_indices, reverse=True):
            del self.trash[i]

        # Player vs Obstacles
        if self.hit_cooldown == 0:
            for obs in self.obstacles:
                dist = np.linalg.norm(self.player_pos - obs['pos'])
                if dist < self.PLAYER_RADIUS + self.OBSTACLE_RADIUS:
                    self.hits += 1
                    self.score -= 5
                    reward -= 5
                    self.hit_cooldown = self.FPS // 2 # 0.5 sec invulnerability
                    # sfx: hit_sound
                    self._create_effect(self.player_pos, self.COLOR_OBSTACLE, 30, 5)
                    break
        return reward
    
    def _create_effect(self, pos, color, count, speed=3):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = np.array([math.cos(angle), math.sin(angle)]) * self.np_random.uniform(1, speed)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(10, 20),
                'color': color
            })

    def _create_bubble(self, at_bottom=False):
        x = self.np_random.uniform(0, self.WORLD_WIDTH)
        y = self.np_random.uniform(0, self.WORLD_HEIGHT) if not at_bottom else self.WORLD_HEIGHT
        return {
            'pos': np.array([x, y]),
            'radius': self.np_random.uniform(1, 4),
            'speed': self.np_random.uniform(0.5, 1.5),
            'sway': self.np_random.uniform(0.01, 0.05)
        }

    def _get_dist_to_nearest_trash(self):
        if not self.trash:
            return 0
        distances = [np.linalg.norm(self.player_pos - item['pos']) for item in self.trash]
        return min(distances)

    def _check_termination(self):
        return not self.trash or self.hits >= self.MAX_HITS or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self._update_camera()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _update_camera(self):
        target_cam_pos = self.player_pos - np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2])
        # Smooth camera movement (lerp)
        self.camera_pos = self.camera_pos * 0.9 + target_cam_pos * 0.1
        # Clamp camera to world boundaries
        self.camera_pos[0] = np.clip(self.camera_pos[0], 0, self.WORLD_WIDTH - self.SCREEN_WIDTH)
        self.camera_pos[1] = np.clip(self.camera_pos[1], 0, self.WORLD_HEIGHT - self.SCREEN_HEIGHT)
    
    def _world_to_screen(self, pos):
        return (pos - self.camera_pos).astype(int)

    def _render_game(self):
        # Background gradient
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        # Seabed
        seabed_y = int(self.WORLD_HEIGHT - self.camera_pos[1])
        if seabed_y < self.SCREEN_HEIGHT:
            pygame.draw.rect(self.screen, self.COLOR_BG_BOTTOM, (0, seabed_y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - seabed_y))

        # Bubbles
        for b in self.bubbles:
            screen_pos = self._world_to_screen(b['pos'])
            pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], int(b['radius']), (255, 255, 255, 50))
        
        # Trash
        for item in self.trash:
            screen_pos = self._world_to_screen(item['pos'])
            # Bobbing animation
            anim_y = 3 * math.sin(self.steps * 0.05 + item['anim_offset'])
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], int(screen_pos[1] + anim_y), self.TRASH_RADIUS, self.COLOR_TRASH)
            pygame.gfxdraw.aacircle(self.screen, screen_pos[0], int(screen_pos[1] + anim_y), self.TRASH_RADIUS, self.COLOR_TRASH)

        # Obstacles
        for obs in self.obstacles:
            screen_pos = self._world_to_screen(obs['pos'])
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], self.OBSTACLE_RADIUS, self.COLOR_OBSTACLE)
            pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], self.OBSTACLE_RADIUS, self.COLOR_OBSTACLE_SPIKE)
            # Spikes
            for i in range(8):
                angle = i * (2 * math.pi / 8) + self.steps * 0.02
                start = (
                    screen_pos[0] + self.OBSTACLE_RADIUS * math.cos(angle),
                    screen_pos[1] + self.OBSTACLE_RADIUS * math.sin(angle)
                )
                end = (
                    screen_pos[0] + (self.OBSTACLE_RADIUS + 4) * math.cos(angle),
                    screen_pos[1] + (self.OBSTACLE_RADIUS + 4) * math.sin(angle)
                )
                pygame.draw.aaline(self.screen, self.COLOR_OBSTACLE_SPIKE, start, end)

        # Player
        player_screen_pos = self._world_to_screen(self.player_pos)
        
        # Glow effect
        glow_surf = pygame.Surface((self.PLAYER_RADIUS * 4, self.PLAYER_RADIUS * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_SUB_GLOW, (self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2), self.PLAYER_RADIUS * 1.5)
        self.screen.blit(glow_surf, (player_screen_pos[0] - self.PLAYER_RADIUS * 2, player_screen_pos[1] - self.PLAYER_RADIUS * 2))

        # Body
        pygame.gfxdraw.filled_ellipse(self.screen, player_screen_pos[0], player_screen_pos[1], self.PLAYER_RADIUS, self.PLAYER_RADIUS - 5, self.COLOR_SUB)
        pygame.gfxdraw.aaellipse(self.screen, player_screen_pos[0], player_screen_pos[1], self.PLAYER_RADIUS, self.PLAYER_RADIUS - 5, self.COLOR_SUB)
        # Window
        pygame.gfxdraw.filled_circle(self.screen, player_screen_pos[0] + 5, player_screen_pos[1], 5, self.COLOR_SUB_WINDOW)
        pygame.gfxdraw.aacircle(self.screen, player_screen_pos[0] + 5, player_screen_pos[1], 5, (255,255,255))
        
        # Particles
        for p in self.particles:
            screen_pos = self._world_to_screen(p['pos'])
            alpha = int(255 * (p['life'] / 20.0))
            color = (*p['color'], alpha)
            size = int(p['life'] / 4)
            if size > 0:
                rect = pygame.Rect(screen_pos[0] - size//2, screen_pos[1] - size//2, size, size)
                shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
                self.screen.blit(shape_surf, rect)

    def _render_ui(self):
        def draw_text(text, pos, font, color, shadow_color):
            shadow = font.render(text, True, shadow_color)
            self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
            surface = font.render(text, True, color)
            self.screen.blit(surface, pos)

        draw_text(f"Score: {self.score}", (10, 10), self.font_main, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        draw_text(f"Trash: {len(self.trash)}/{self.NUM_TRASH}", (self.SCREEN_WIDTH - 150, 10), self.font_main, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Hits indicator
        hit_text = f"Damage: {self.hits}/{self.MAX_HITS}"
        hit_color = self.COLOR_TEXT if self.hits < self.MAX_HITS -1 else (255, 100, 100)
        draw_text(hit_text, (10, self.SCREEN_HEIGHT - 30), self.font_small, hit_color, self.COLOR_TEXT_SHADOW)
        
        if self.game_over:
            msg = "MISSION COMPLETE!" if not self.trash else "MISSION FAILED"
            draw_text(msg, (self.SCREEN_WIDTH // 2 - 150, self.SCREEN_HEIGHT // 2 - 30), self.font_main, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "hits": self.hits, "trash_remaining": len(self.trash)}

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Submersible Trash Collector")
    
    running = True
    total_reward = 0
    
    # Main game loop
    while running:
        movement_action = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Wait 3 seconds
            obs, info = env.reset()
            total_reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    
    env.close()