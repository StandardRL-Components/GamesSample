
# Generated: 2025-08-27T22:11:17.750055
# Source Brief: brief_03034.md
# Brief Index: 3034

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import os
import pygame


class GameEnv(gym.Env):
    """
    A minimalist platformer where the only control is jump, challenging players
    to precisely time leaps across procedurally generated platforms to reach the flag.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Press space to jump. Time your jumps to land on the platforms and ascend."
    )

    game_description = (
        "A minimalist vertical platformer. Jump from platform to platform to reach the flag at the top."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_STEPS = 1000
        self.NUM_PLATFORMS = 50

        # Colors
        self.COLOR_BG_TOP = (135, 206, 235)
        self.COLOR_BG_BOTTOM = (70, 130, 180)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLATFORM = (100, 100, 100)
        self.COLOR_FLAG_POLE = (192, 192, 192)
        self.COLOR_FLAG = (0, 200, 0)
        self.COLOR_PARTICLE = (200, 200, 200)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (0, 0, 0)

        # Physics
        self.GRAVITY = 0.4
        self.JUMP_STRENGTH = 10.5
        self.PLAYER_WIDTH = 20
        self.PLAYER_HEIGHT = 20
        self.TERMINAL_VELOCITY = 15

        # --- Gymnasium Setup ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        
        self.player_pos = [0, 0]
        self.player_vel = [0, 0]
        self.on_ground = False
        self.last_space_held = False
        
        self.platforms = []
        self.flag_rect = None
        self.particles = []
        
        self.camera_y = 0
        self.start_y = 0
        self.last_platform_landed_idx = -1

        # Final validation check
        # Note: reset() is called by some gym wrappers automatically.
        # Calling it here ensures state is valid for validation.
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.start_y = self.SCREEN_HEIGHT * 0.8
        self.player_pos = [
            self.SCREEN_WIDTH / 2 - self.PLAYER_WIDTH / 2,
            self.start_y
        ]
        self.player_vel = [0, 0]
        self.on_ground = True
        self.last_space_held = False
        
        self.particles = []
        self.last_platform_landed_idx = 0

        self._generate_level()
        
        self.camera_y = self.player_pos[1] - self.SCREEN_HEIGHT * 0.75
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms = []
        
        start_plat_width = 120
        start_plat = pygame.Rect(
            (self.SCREEN_WIDTH / 2) - start_plat_width / 2,
            self.start_y + self.PLAYER_HEIGHT,
            start_plat_width,
            20
        )
        self.platforms.append(start_plat)
        
        current_y = start_plat.y
        
        max_jump_height = (self.JUMP_STRENGTH**2) / (2 * self.GRAVITY)
        
        for _ in range(self.NUM_PLATFORMS):
            width = self.np_random.uniform(60, 150)
            x = (self.SCREEN_WIDTH / 2) - (width / 2)
            dy = self.np_random.uniform(30, max_jump_height * 0.9)
            y = current_y - dy
            
            new_plat = pygame.Rect(x, y, width, 20)
            self.platforms.append(new_plat)
            current_y = y
            
        last_plat = self.platforms[-1]
        self.flag_rect = pygame.Rect(last_plat.centerx - 2, last_plat.y - 40, 4, 40)

    def step(self, action):
        reward = -0.01  # Time penalty

        # --- 1. Handle Input ---
        space_held = action[1] == 1
        jump_action = space_held and not self.last_space_held
        self.last_space_held = space_held
        
        # --- 2. Update Physics ---
        old_player_bottom = self.player_pos[1] + self.PLAYER_HEIGHT
        
        if jump_action and self.on_ground:
            self.player_vel[1] = -self.JUMP_STRENGTH
            self.on_ground = False
            # SFX: Jump sound

        self.player_vel[1] += self.GRAVITY
        self.player_vel[1] = min(self.player_vel[1], self.TERMINAL_VELOCITY)
        
        self.player_pos[1] += self.player_vel[1]
        
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        
        # --- 3. Collision Detection ---
        self.on_ground = False
        if self.player_vel[1] >= 0:
            for i, plat in enumerate(self.platforms):
                if player_rect.colliderect(plat) and old_player_bottom <= plat.top + 1:
                    player_rect.bottom = plat.top
                    self.player_pos[1] = player_rect.y
                    self.player_vel[1] = 0
                    self.on_ground = True
                    
                    if self.last_platform_landed_idx != i:
                        reward += 1.0
                        self.score += 10
                        self.last_platform_landed_idx = i
                        self._create_landing_particles(player_rect.midbottom)
                        # SFX: Land sound
                    break
        
        # --- 4. Update Game Logic ---
        self._update_particles()
        
        target_camera_y = self.player_pos[1] - self.SCREEN_HEIGHT * 0.6
        self.camera_y += (target_camera_y - self.camera_y) * 0.1
        
        # --- 5. Calculate Rewards & Termination ---
        if self.player_pos[1] < self.start_y:
            reward += 0.1

        terminated = False
        
        if self.flag_rect and player_rect.colliderect(self.flag_rect):
            reward += 100
            self.score += 1000
            terminated = True
            # SFX: Win sound
            
        if self.player_pos[1] > self.camera_y + self.SCREEN_HEIGHT + self.PLAYER_HEIGHT:
            reward -= 10
            terminated = True
            # SFX: Fall sound
            
        if self.steps >= self.MAX_STEPS - 1:
            terminated = True
            
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            r = int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio)
            g = int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio)
            b = int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Draw platforms
        for plat in self.platforms:
            if plat.bottom > self.camera_y and plat.top < self.camera_y + self.SCREEN_HEIGHT:
                draw_rect = plat.move(0, -self.camera_y)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, draw_rect, border_radius=3)
                
        # Draw flag
        if self.flag_rect:
            if self.flag_rect.bottom > self.camera_y and self.flag_rect.top < self.camera_y + self.SCREEN_HEIGHT:
                pole_rect = self.flag_rect.move(0, -self.camera_y)
                pygame.draw.rect(self.screen, self.COLOR_FLAG_POLE, pole_rect)
                flag_points = [
                    (pole_rect.right, pole_rect.top),
                    (pole_rect.right + 20, pole_rect.top + 10),
                    (pole_rect.right, pole_rect.top + 20)
                ]
                pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_points)

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1] - self.camera_y))
            alpha = max(0, min(255, int(255 * (p['life'] / 30))))
            color = (*self.COLOR_PARTICLE, alpha)
            
            particle_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.rect(particle_surf, color, (0, 0, 3, 3))
            self.screen.blit(particle_surf, (pos[0]-2, pos[1]-2))

        # Draw player
        player_draw_pos = (
            int(self.player_pos[0]),
            int(self.player_pos[1] - self.camera_y)
        )
        player_rect = pygame.Rect(player_draw_pos, (self.PLAYER_WIDTH, self.PLAYER_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_ui(self):
        def draw_text_with_shadow(text, pos, font, color, shadow_color):
            shadow_text = font.render(text, True, shadow_color)
            main_text = font.render(text, True, color)
            self.screen.blit(shadow_text, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(main_text, pos)

        score_str = f"SCORE: {self.score}"
        draw_text_with_shadow(score_str, (10, 10), self.font, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        steps_str = f"STEPS: {self.steps}/{self.MAX_STEPS}"
        text_width = self.font.size(steps_str)[0]
        draw_text_with_shadow(steps_str, (self.SCREEN_WIDTH - text_width - 10, 10), self.font, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

    def _create_landing_particles(self, pos):
        for _ in range(10):
            angle = self.np_random.uniform(math.pi * 1.1, math.pi * 1.9)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifespan})

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += self.GRAVITY * 0.1
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.font.quit()
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
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Prevents Pygame from opening a window if not needed
    
    # To play with a window, comment out the line above and uncomment the one below
    # Also, change render_mode to "human" in the GameEnv constructor if a human render mode is implemented
    
    # For this example, we will render to a window to test visually
    use_human_player = True
    if use_human_player:
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc. depending on your OS
        
    env = GameEnv()
    obs, info = env.reset()
    
    if use_human_player:
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Minimalist Jumper")
        
    terminated = False
    total_reward = 0
    
    # --- Action mapping for human player ---
    # action = [movement, space, shift]
    action = [0, 0, 0]

    while not terminated:
        if use_human_player:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")

            keys = pygame.key.get_pressed()
            action[1] = 1 if keys[pygame.K_SPACE] else 0
            
            # Render to the display window
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
        else: # AI/Random agent
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            if use_human_player:
                # Wait for reset command
                pass
            else:
                obs, info = env.reset()
                total_reward = 0

        if use_human_player:
            env.clock.tick(30) # Control the frame rate

    env.close()