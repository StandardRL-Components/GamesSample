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


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Arrow keys to jump. Hold Space for a powerful jump, or Shift for a shorter, precise jump."
    )

    # Short, user-facing description of the game
    game_description = (
        "Hop between procedurally generated platforms to reach the top before the timer runs out."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and timing
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 10
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (20, 40, 80)
        self.COLOR_PLATFORM = (100, 150, 255)
        self.COLOR_PLATFORM_TOP = (150, 200, 255)
        self.COLOR_GOAL_PLATFORM = (255, 200, 0)
        self.COLOR_GOAL_PLATFORM_TOP = (255, 255, 100)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (0, 0, 0, 128)

        # Physics and game parameters
        self.GRAVITY = 0.5
        self.PLAYER_SIZE = np.array([16, 16])
        self.BASE_JUMP_POWER_V = -9
        self.BASE_JUMP_POWER_H = 6
        self.SUPER_JUMP_MODIFIER = 1.4
        self.PRECISION_JUMP_MODIFIER = 0.6
        self.PLATFORM_HEIGHT = 12
        self.NUM_PLATFORMS = 12
        
        # Difficulty scaling
        self.total_jumps_ever = 0
        self.PLATFORM_DIFFICULTY_STEP = 500
        self.platform_base_width = 120
        self.platform_min_width = 40

        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.is_grounded = None
        self.player_squash = None
        self.platforms = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        self.terminated = False
        self.current_platform_idx = 0
        self.highest_platform_idx = 0
        self.goal_platform_idx = 0


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset core state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        self.terminated = False

        # Player state
        self.player_pos = np.array([self.WIDTH / 2.0, self.HEIGHT - 60.0])
        self.player_vel = np.array([0.0, 0.0])
        self.is_grounded = True
        self.player_squash = 0.0
        
        # Game elements
        self._generate_platforms()
        self.current_platform_idx = 0
        self.highest_platform_idx = 0
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, self.terminated, False, self._get_info()

        self.steps += 1
        self.reward_this_step = 0

        self._handle_input(action)
        self._update_player()
        self._update_particles()
        
        self.terminated = self._check_termination()
        if self.terminated:
            self.game_over = True
            
        reward = self.reward_this_step
        self.score += reward
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            self.terminated,
            truncated,
            self._get_info(),
        )

    def _generate_platforms(self):
        self.platforms = []
        
        # Starting platform
        start_plat_width = 160
        start_plat = pygame.Rect(
            self.WIDTH / 2 - start_plat_width / 2,
            self.HEIGHT - 40,
            start_plat_width,
            self.PLATFORM_HEIGHT,
        )
        self.platforms.append(start_plat)

        # Procedurally generate intermediate platforms
        y_spacing = (self.HEIGHT - 120) / (self.NUM_PLATFORMS - 1)
        max_h_dist = 180
        
        # Calculate current platform width based on difficulty
        difficulty_reduction = (self.total_jumps_ever // self.PLATFORM_DIFFICULTY_STEP)
        current_max_width = max(self.platform_min_width, self.platform_base_width - difficulty_reduction * 5)
        current_min_width = max(self.platform_min_width, 80 - difficulty_reduction * 5)


        for i in range(1, self.NUM_PLATFORMS):
            prev_plat = self.platforms[i - 1]
            y = self.HEIGHT - 40 - (i * y_spacing) - self.np_random.uniform(0, 20)
            
            x_offset = self.np_random.uniform(-max_h_dist, max_h_dist)
            x = prev_plat.centerx + x_offset
            
            width = self.np_random.integers(low=current_min_width, high=current_max_width + 1)
            
            # Clamp to screen
            x = np.clip(x - width / 2, 20, self.WIDTH - width - 20)

            new_plat = pygame.Rect(x, y, width, self.PLATFORM_HEIGHT)
            self.platforms.append(new_plat)
            
        # The last platform is the goal
        self.goal_platform_idx = len(self.platforms) - 1

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if self.is_grounded and movement != 0:
            jump_power_v = self.BASE_JUMP_POWER_V
            jump_power_h = self.BASE_JUMP_POWER_H
            
            # Apply modifiers
            if space_held:
                jump_power_v *= self.SUPER_JUMP_MODIFIER
                jump_power_h *= self.SUPER_JUMP_MODIFIER
            elif shift_held:
                jump_power_v *= self.PRECISION_JUMP_MODIFIER
                jump_power_h *= self.PRECISION_JUMP_MODIFIER
            
            # Set velocity based on direction
            if movement == 1: # Up
                self.player_vel = np.array([0, jump_power_v])
            elif movement == 2: # Down
                self.player_vel = np.array([0, -jump_power_v * 0.3]) # Small hop down
            elif movement == 3: # Left
                self.player_vel = np.array([-jump_power_h, jump_power_v * 0.7])
            elif movement == 4: # Right
                self.player_vel = np.array([jump_power_h, jump_power_v * 0.7])

            self.is_grounded = False
            self.player_squash = 1.0  # Stretch on jump
            self._create_jump_particles(10)

    def _update_player(self):
        # Animate squash and stretch
        self.player_squash *= 0.85

        if self.is_grounded:
            self.reward_this_step += 0.01 # Small reward for being stable
            return

        # Apply gravity
        prev_pos_y = self.player_pos[1]
        self.player_vel[1] += self.GRAVITY
        self.player_pos += self.player_vel
        
        # Clamp horizontal position
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH - self.PLAYER_SIZE[0])

        # Check for landing
        if self.player_vel[1] > 0: # Only check for landing if falling
            player_rect = pygame.Rect(self.player_pos, self.PLAYER_SIZE)
            for i, plat in enumerate(self.platforms):
                if player_rect.colliderect(plat):
                    # Check if player was above the platform in the previous frame
                    if prev_pos_y + self.PLAYER_SIZE[1] <= plat.top:
                        self.player_pos[1] = plat.top - self.PLAYER_SIZE[1]
                        self.player_vel = np.array([0.0, 0.0])
                        self.is_grounded = True
                        self.player_squash = -1.0 # Squash on land
                        self._create_land_particles(15)

                        # Landing rewards
                        if i > self.highest_platform_idx:
                            self.reward_this_step += 1.0 * (i - self.highest_platform_idx)
                            self.highest_platform_idx = i
                        elif i < self.current_platform_idx:
                            self.reward_this_step -= 0.5
                        
                        self.current_platform_idx = i
                        self.total_jumps_ever += 1
                        
                        # Check for victory
                        if i == self.goal_platform_idx:
                            self.reward_this_step += 100
                            self.terminated = True
                        
                        break

    def _update_particles(self):
        updated_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'][1] += 0.1  # Particle gravity
            p['life'] -= 1
            if p['life'] > 0:
                updated_particles.append(p)
        self.particles = updated_particles

    def _check_termination(self):
        if self.terminated: # Already terminated by goal
            return True
            
        # Fell off screen
        if self.player_pos[1] > self.HEIGHT:
            self.reward_this_step = -100
            return True
            
        # Timer ran out (this is now handled by truncation)
        # if self.steps >= self.MAX_STEPS:
        #     return True
            
        return False

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS,
            "highest_platform": self.highest_platform_idx,
        }

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = max(0, p['life'] / p['max_life'])
            color = (p['color'][0], p['color'][1], p['color'][2], int(alpha * 255))
            size = int(p['size'] * alpha)
            if size > 0:
                # Using a rect for a square particle
                rect = pygame.Rect(int(p['pos'][0]), int(p['pos'][1]), size, size)
                pygame.draw.rect(self.screen, color, rect)

        # Render platforms
        for i, plat in enumerate(self.platforms):
            is_goal = (i == self.goal_platform_idx)
            main_color = self.COLOR_GOAL_PLATFORM if is_goal else self.COLOR_PLATFORM
            top_color = self.COLOR_GOAL_PLATFORM_TOP if is_goal else self.COLOR_PLATFORM_TOP
            
            pygame.draw.rect(self.screen, main_color, plat)
            pygame.draw.rect(self.screen, top_color, (plat.x, plat.y, plat.width, 3))
            
            # Add a subtle glow
            if i == self.current_platform_idx and self.is_grounded:
                glow_rect = plat.inflate(6, 6)
                s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(s, (255, 255, 255, 30), s.get_rect(), border_radius=5)
                self.screen.blit(s, glow_rect.topleft)

        # Render player with squash and stretch
        squash_factor = 0.4 * self.player_squash
        visual_h = self.PLAYER_SIZE[1] * (1 + squash_factor)
        visual_w = self.PLAYER_SIZE[0] * (1 - squash_factor)
        
        player_render_pos = (
            self.player_pos[0] + (self.PLAYER_SIZE[0] - visual_w) / 2,
            self.player_pos[1] + (self.PLAYER_SIZE[1] - visual_h) / 2
        )
        player_rect = pygame.Rect(player_render_pos, (visual_w, visual_h))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {int(self.score)}"
        self._draw_text(score_text, (10, 10), self.font_small, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        timer_text = f"TIME: {max(0, time_left):.2f}"
        color = (255, 100, 100) if time_left < 3 else self.COLOR_TEXT
        self._draw_text(timer_text, (self.WIDTH - 10, 10), self.font_small, color, self.COLOR_TEXT_SHADOW, align="topright")
        
        # Game over message
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 128))
            self.screen.blit(s, (0, 0))
            
            msg = "YOU WIN!" if self.current_platform_idx == self.goal_platform_idx else "GAME OVER"
            self._draw_text(msg, (self.WIDTH/2, self.HEIGHT/2 - 20), self.font_large, self.COLOR_PLAYER, self.COLOR_TEXT_SHADOW, align="center")

    def _draw_text(self, text, pos, font, color, shadow_color, align="topleft"):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        shadow_surf = font.render(text, True, shadow_color)
        
        if align == "topleft":
            text_rect.topleft = pos
        elif align == "topright":
            text_rect.topright = pos
        elif align == "center":
            text_rect.center = pos
        
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _create_land_particles(self, count):
        for _ in range(count):
            self.particles.append({
                'pos': self.player_pos.copy() + self.PLAYER_SIZE / 2,
                'vel': np.array([self.np_random.uniform(-2, 2), self.np_random.uniform(-3, -1)]),
                'life': self.np_random.integers(15, 25),
                'max_life': 25,
                'color': (255, 255, 100),
                'size': self.np_random.integers(2, 5)
            })

    def _create_jump_particles(self, count):
        for _ in range(count):
            self.particles.append({
                'pos': self.player_pos.copy() + self.PLAYER_SIZE / 2,
                'vel': np.array([self.np_random.uniform(-1, 1), self.np_random.uniform(1, 2)]),
                'life': self.np_random.integers(10, 20),
                'max_life': 20,
                'color': (200, 200, 255),
                'size': self.np_random.integers(2, 4)
            })

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Platform Jumper")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}")
            # In a real scenario, you'd reset here. We'll just let the game over screen show.
            # obs, info = env.reset()
            # total_reward = 0
            
        clock.tick(env.FPS)
        
    env.close()