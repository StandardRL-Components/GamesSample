import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to run, ↑ to jump. Avoid red platforms and orange saws. Reach the flag at the end!"
    )

    game_description = (
        "A fast-paced, retro-futuristic platformer. Guide your robot through a "
        "procedurally generated neon world, jumping over pits and dodging deadly saws to reach the finish line."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH = 640
        self.HEIGHT = 400
        self.LEVEL_LENGTH = 10000  # pixels

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)

        # Colors
        self.COLOR_BG = (15, 15, 35)
        self.COLOR_GRID = (30, 30, 60)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 150)
        self.COLOR_PLATFORM_SAFE = (0, 100, 255)
        self.COLOR_PLATFORM_CRUMBLE = (255, 50, 50)
        self.COLOR_SAW = (255, 128, 0)
        self.COLOR_SAW_GLOW = (200, 100, 0)
        self.COLOR_FINISH = (255, 255, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PARTICLE_JUMP = (0, 255, 255, 150)
        self.COLOR_PARTICLE_DEATH = (255, 50, 50, 200)
        
        # Physics constants
        self.GRAVITY = 0.5
        self.PLAYER_ACCEL = 0.8
        self.PLAYER_FRICTION = 0.85
        self.PLAYER_JUMP_STRENGTH = -10.5
        self.MAX_VEL_X = 6
        self.MAX_VEL_Y = 15

        # Game state variables
        self.player_pos = None
        self.player_vel = None
        self.player_size = None
        self.on_ground = None
        self.camera_x = None
        self.platforms = None
        self.saws = None
        self.particles = None
        self.generation_pointer = None
        self.difficulty_tier = None
        
        self.steps = 0
        self.score = 0
        self.lives = 0
        self._prev_lives = 0
        self.time_limit = 0
        self.game_over = False
        self.last_checkpoint_x = 0

        # self.reset() is called here to ensure all state variables are initialized
        # before any other methods like `validate_implementation` are called.
        # This prevents AttributeError for variables initialized only in `reset`.
        if render_mode != "human": # Avoid calling reset if not needed for validation
            self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.lives = 3
        self._prev_lives = self.lives
        self.time_limit = 60 * 30  # 60 seconds at 30fps
        self.game_over = False
        self.last_checkpoint_x = 0
        self.difficulty_tier = 0

        self.player_size = pygame.Vector2(24, 32)
        self.player_pos = pygame.Vector2(150, 200)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False

        self.camera_x = 0
        self.platforms = []
        self.saws = []
        self.particles = []
        
        # Procedural generation setup
        self.generation_pointer = 0
        self._generate_initial_chunk()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        self.time_limit -= 1

        # 1. Handle Input
        movement, _, _ = action
        self._handle_input(movement)
        
        prev_player_x = self.player_pos.x

        # 2. Update Game Logic
        self._update_player()
        self._update_world()
        self._handle_collisions()
        self._update_camera()
        self._procedural_generation()
        
        # 3. Calculate Reward
        reward += self._calculate_reward(prev_player_x)
        self.score += reward

        # 4. Check Termination
        terminated = self._check_termination()
        if terminated:
            self.game_over = True

        # Tick clock for auto-advance
        self.clock.tick(30)
        
        truncated = self.steps >= 5000

        return self._get_observation(), float(reward), terminated, truncated, self._get_info()

    def _handle_input(self, movement):
        # movement: 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1 and self.on_ground: # Jump
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH
            self.on_ground = False
            # Sound: jump.wav
            for _ in range(10):
                self.particles.append(self._create_particle(self.player_pos + (self.player_size.x/2, self.player_size.y), self.COLOR_PARTICLE_JUMP, 20))
        if movement == 3: # Left
            self.player_vel.x -= self.PLAYER_ACCEL
        if movement == 4: # Right
            self.player_vel.x += self.PLAYER_ACCEL

    def _update_player(self):
        # Apply gravity
        self.player_vel.y += self.GRAVITY
        
        # Apply friction
        self.player_vel.x *= self.PLAYER_FRICTION
        
        # Clamp velocity
        self.player_vel.x = max(-self.MAX_VEL_X, min(self.MAX_VEL_X, self.player_vel.x))
        self.player_vel.y = max(-self.MAX_VEL_Y, min(self.MAX_VEL_Y, self.player_vel.y))
        
        # Update position
        self.player_pos += self.player_vel

        # Platform collisions
        self.on_ground = False
        player_rect = pygame.Rect(self.player_pos, self.player_size)
        
        for p_rect, p_data in self.platforms:
            if player_rect.colliderect(p_rect):
                # Check for vertical collision (landing on top)
                if self.player_vel.y > 0 and player_rect.bottom - self.player_vel.y <= p_rect.top + 1:
                    self.player_pos.y = p_rect.top - self.player_size.y
                    self.player_vel.y = 0
                    self.on_ground = True
                    if p_data['type'] == 'crumble' and p_data['timer'] is None:
                        p_data['timer'] = 15 # 0.5s at 30fps
                        p_data['reward_given'] = False # for reward calculation
                    # Sound: land.wav
                # Horizontal collision
                elif self.player_vel.x > 0 and player_rect.right - self.player_vel.x <= p_rect.left:
                    self.player_pos.x = p_rect.left - self.player_size.x
                    self.player_vel.x = 0
                elif self.player_vel.x < 0 and player_rect.left - self.player_vel.x >= p_rect.right:
                    self.player_pos.x = p_rect.right
                    self.player_vel.x = 0
        
        # Check for falling out of the world
        if self.player_pos.y > self.HEIGHT + 50:
            self._player_die()

    def _update_world(self):
        # Update crumbling platforms
        for _, p_data in self.platforms:
            if p_data['timer'] is not None:
                p_data['timer'] -= 1
                if p_data['timer'] < 0:
                    p_data['active'] = False # Mark for removal
        self.platforms = [p for p in self.platforms if p[1]['active']]

        # Update saws
        difficulty_mod = 1.0 + (self.difficulty_tier * 0.05)
        for saw in self.saws:
            saw['angle'] = (saw['angle'] + saw['rot_speed']) % 360
            saw['pos'] += saw['vel'] * difficulty_mod
            if saw['pos'].x < saw['path_start_x'] or saw['pos'].x > saw['path_end_x']:
                saw['vel'].x *= -1

        # Update particles
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['vel'].y += 0.1 # particle gravity

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos, self.player_size)
        player_center = player_rect.center
        for saw in self.saws:
            dist = pygame.Vector2(player_center).distance_to(saw['pos'])
            if dist < saw['radius'] + (player_rect.width / 4): # smaller hitbox
                self._player_die()
                # Sound: player_hit.wav
                return

    def _player_die(self):
        if self.lives > 0: # Prevent multiple deaths in one frame
            self.lives -= 1
            # Sound: death_explosion.wav
            for _ in range(30):
                self.particles.append(self._create_particle(self.player_pos + self.player_size/2, self.COLOR_PARTICLE_DEATH, 40, 2))

            if self.lives > 0:
                # Respawn at the start of the current screen view
                self.player_pos = pygame.Vector2(self.camera_x + 50, self.HEIGHT / 2)
                self.player_vel = pygame.Vector2(0, 0)
            else:
                self.game_over = True

    def _update_camera(self):
        target_camera_x = self.player_pos.x - self.WIDTH / 3
        # Smooth camera follow
        self.camera_x += (target_camera_x - self.camera_x) * 0.1
        self.camera_x = max(0, min(self.LEVEL_LENGTH - self.WIDTH, self.camera_x))

    def _procedural_generation(self):
        # Update difficulty
        new_tier = self.steps // 500
        if new_tier > self.difficulty_tier:
            self.difficulty_tier = new_tier
            # Sound: difficulty_up.wav

        while self.generation_pointer < self.camera_x + self.WIDTH + 200:
            last_platform = self.platforms[-1][0] if self.platforms else pygame.Rect(0, 350, 200, 50)
            
            # Anti-softlock: ensure gaps are jumpable
            max_gap = 120 + abs(self.player_vel.x) * 5
            gap_x = self.np_random.integers(40, max_gap)
            delta_y = self.np_random.integers(-80, 80)
            
            new_x = last_platform.right + gap_x
            new_y = np.clip(last_platform.y + delta_y, 150, self.HEIGHT - 50)
            new_width = self.np_random.integers(80, 250)
            
            # Make sure we don't go past the level end
            if new_x + new_width > self.LEVEL_LENGTH - 200:
                break

            # Add platform
            p_type = 'crumble' if self.np_random.random() < 0.2 + (self.difficulty_tier * 0.05) else 'safe'
            self._add_platform(new_x, new_y, new_width, 50, p_type)
            
            # Maybe add a saw
            if self.steps > 50 and self.np_random.random() < 0.25 + (self.difficulty_tier * 0.05):
                saw_x = new_x + new_width / 2
                saw_y = new_y - 30
                self._add_saw(saw_x, saw_y)

            self.generation_pointer = new_x + new_width
            
    def _generate_initial_chunk(self):
        # Safe starting platform
        self._add_platform(50, 300, 200, 100, 'safe')
        self.generation_pointer = 250
        # Ensure a few safe platforms to start
        for i in range(5):
            last_platform = self.platforms[-1][0]
            new_x = last_platform.right + self.np_random.integers(50, 100)
            new_y = np.clip(last_platform.y + self.np_random.integers(-20, 20), 250, 350)
            new_width = self.np_random.integers(100, 150)
            self._add_platform(new_x, new_y, new_width, 50, 'safe')
            self.generation_pointer = new_x + new_width

    def _add_platform(self, x, y, w, h, p_type):
        rect = pygame.Rect(x, y, w, h)
        data = {'type': p_type, 'timer': None, 'active': True}
        self.platforms.append((rect, data))

    def _add_saw(self, x, y):
        path_width = self.np_random.uniform(50, 150)
        saw = {
            'pos': pygame.Vector2(x, y),
            'radius': self.np_random.uniform(15, 25),
            'angle': 0,
            'rot_speed': self.np_random.uniform(5, 10) * self.np_random.choice([-1, 1]),
            'path_start_x': x - path_width / 2,
            'path_end_x': x + path_width / 2,
            'vel': pygame.Vector2(self.np_random.uniform(1, 2.5), 0)
        }
        self.saws.append(saw)

    def _calculate_reward(self, prev_player_x):
        r = 0
        # Reward for moving right, penalize moving left
        progress = self.player_pos.x - prev_player_x
        if progress > 0:
            r += 0.1 * progress / self.MAX_VEL_X
        else:
            r += 0.01 * progress / self.MAX_VEL_X # Negative progress is penalized

        # Penalty for touching crumbling platform
        player_rect = pygame.Rect(self.player_pos, self.player_size)
        for p_rect, p_data in self.platforms:
            if p_data['type'] == 'crumble' and p_data['timer'] is not None and not p_data.get('reward_given', False):
                if player_rect.colliderect(p_rect):
                    r -= 0.1
                    p_data['reward_given'] = True

        # Checkpoint reward
        checkpoint_dist = 1000
        if self.player_pos.x > self.last_checkpoint_x + checkpoint_dist:
            self.last_checkpoint_x += checkpoint_dist
            r += 1.0

        # Penalty for losing a life
        if self.lives < self._prev_lives:
            r -= 5.0
        self._prev_lives = self.lives

        # Goal reward
        if self.player_pos.x >= self.LEVEL_LENGTH - 100:
            r += 100.0
            
        return r

    def _check_termination(self):
        if self.lives <= 0:
            return True
        if self.time_limit <= 0:
            return True
        if self.player_pos.x >= self.LEVEL_LENGTH - 100:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "time_remaining": self.time_limit / 30,
            "player_x": self.player_pos.x,
        }

    def _render_game(self):
        # Draw background grid (parallax)
        for i in range(0, self.WIDTH, 50):
            x = i - (self.camera_x * 0.1 % 50)
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 50):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)

        # Draw particles
        for p in self.particles:
            alpha_color = (*p['color'][:3], int(p['color'][3] * (p['lifespan'] / p['max_lifespan'])))
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, alpha_color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'].x - self.camera_x - p['size']), int(p['pos'].y - p['size'])), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw platforms
        for p_rect, p_data in self.platforms:
            draw_rect = p_rect.move(-self.camera_x, 0)
            color = self.COLOR_PLATFORM_SAFE if p_data['type'] == 'safe' else self.COLOR_PLATFORM_CRUMBLE
            if p_data['timer'] is not None:
                # Shake effect
                if p_data['timer'] > 0:
                    draw_rect.x += self.np_random.uniform(-2, 2)
                    draw_rect.y += self.np_random.uniform(-2, 2)
            pygame.draw.rect(self.screen, color, draw_rect, border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), draw_rect, width=2, border_radius=3)

        # Draw saws
        for saw in self.saws:
            pos = (int(saw['pos'].x - self.camera_x), int(saw['pos'].y))
            self._draw_glowing_circle(self.screen, pos, int(saw['radius']), self.COLOR_SAW, self.COLOR_SAW_GLOW)
            for i in range(8):
                angle = math.radians(saw['angle'] + i * 45)
                start = (pos[0] + math.cos(angle) * saw['radius'] * 0.3, pos[1] + math.sin(angle) * saw['radius'] * 0.3)
                end = (pos[0] + math.cos(angle) * saw['radius'], pos[1] + math.sin(angle) * saw['radius'])
                pygame.draw.line(self.screen, self.COLOR_SAW_GLOW, start, end, 2)

        # Draw finish line
        finish_x = self.LEVEL_LENGTH - 100 - self.camera_x
        pygame.draw.rect(self.screen, self.COLOR_FINISH, (finish_x, 0, 10, self.HEIGHT))
        pygame.draw.polygon(self.screen, self.COLOR_FINISH, [(finish_x+10, 20), (finish_x+60, 45), (finish_x+10, 70)])

        # Draw player
        player_rect = pygame.Rect(
            int(self.player_pos.x - self.camera_x), int(self.player_pos.y),
            self.player_size.x, self.player_size.y
        )
        # Simple running animation
        if self.on_ground and abs(self.player_vel.x) > 0.5:
             player_rect.height = self.player_size.y - 4 + 4 * math.sin(self.steps * 0.5)
        self._draw_glowing_rect(self.screen, player_rect, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)
        
        # Jetpack effect
        if not self.on_ground and self.player_vel.y < 0:
            for i in range(3):
                p_pos = (player_rect.centerx, player_rect.bottom + i * 4)
                p_size = 8 - i * 2
                self.particles.append(self._create_particle(p_pos, (255, 165, 0, 100), 5, p_size/2, vel_y=1))


    def _render_ui(self):
        # Lives
        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (10, 10))

        # Timer
        time_str = f"TIME: {int(self.time_limit / 30):02d}"
        time_text = self.font_small.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))
        
        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "LEVEL COMPLETE!" if self.player_pos.x >= self.LEVEL_LENGTH - 100 else "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_FINISH if msg.startswith("LEVEL") else self.COLOR_PLATFORM_CRUMBLE)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _create_particle(self, pos, color, lifespan, size_mult=1, vel_y=0):
        vel = pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-3, 0) + vel_y)
        size = self.np_random.uniform(2, 5) * size_mult
        return {'pos': pygame.Vector2(pos), 'vel': vel, 'lifespan': lifespan, 'max_lifespan': lifespan, 'color': color, 'size': size}

    def _draw_glowing_circle(self, surface, pos, radius, color, glow_color):
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(radius * 1.5), glow_color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(radius * 1.2), glow_color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)

    def _draw_glowing_rect(self, surface, rect, color, glow_color):
        glow_rect = rect.inflate(10, 10)
        shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, (*glow_color, 100), (0, 0, *glow_rect.size), border_radius=8)
        inner_glow_rect = pygame.Rect(3, 3, glow_rect.width - 6, glow_rect.height - 6)
        pygame.draw.rect(shape_surf, (*glow_color, 150), inner_glow_rect, border_radius=6)
        surface.blit(shape_surf, glow_rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)
        pygame.draw.rect(surface, color, rect, border_radius=4)

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    print("Environment created and reset successfully.")
    print("User Guide:", env.user_guide)
    print("Game Description:", env.game_description)
    print("Initial Info:", info)
    
    # Run for a few steps with random actions
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (i + 1) % 20 == 0:
            print(f"Step {i+1}: Reward={reward:.2f}, Total Reward={total_reward:.2f}, Info={info}")
        if terminated or truncated:
            print("Episode finished.")
            obs, info = env.reset()
            
    env.close()
    print("Environment closed.")