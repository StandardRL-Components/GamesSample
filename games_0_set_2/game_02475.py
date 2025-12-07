import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Press space to jump. Time your jumps to cross the gaps and reach the green flag."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist platformer about precise timing. Navigate through three challenging stages by jumping across increasingly difficult gaps."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Headless Setup ---
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLATFORM = (100, 110, 120)
        self.COLOR_GOAL = (0, 200, 100)
        self.COLOR_DANGER = (255, 50, 50)
        self.COLOR_PARTICLE = (220, 220, 220)
        self.COLOR_UI = (200, 200, 210)

        # --- Game Constants ---
        self.GRAVITY = 0.3
        self.JUMP_STRENGTH = -8
        self.PLAYER_SPEED = 3.0
        self.PLAYER_SIZE = 20
        self.MAX_STEPS = 3000
        self.NUM_STAGES = 3

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.terminated = False
        self.stage = 1
        self.stage_timer = 0

        self.player_pos = [0.0, 0.0]
        self.player_vel_y = 0.0
        self.player_on_ground = False
        self.just_landed_reward_given = False
        
        self.camera_offset_x = 0.0
        self.platforms = []
        self.end_flag = None
        self.particles = []
        
        self.death_flash_alpha = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.terminated = False
        self.stage = 1
        self.stage_timer = 0

        self.player_pos = [100.0, 200.0]
        self.player_vel_y = 0.0
        self.player_on_ground = False
        self.just_landed_reward_given = True # Don't give reward for initial spawn

        self.particles.clear()
        self._generate_stage(self.stage)
        self.camera_offset_x = self.player_pos[0] - 150 # Start with player on left side

        self.death_flash_alpha = 0
        
        return self._get_observation(), self._get_info()

    def _generate_stage(self, stage_num):
        self.platforms.clear()
        
        base_gap = 80
        base_platform_width = 200
        
        current_x = -self.screen_width # Start off-screen
        
        # Starting platform
        start_platform = pygame.Rect(current_x, 300, self.screen_width * 2, 100)
        self.platforms.append(start_platform)
        current_x += start_platform.width
        
        num_platforms = 15 + stage_num * 5

        for i in range(num_platforms):
            gap_multiplier = 1.0 + 0.1 * (stage_num - 1)
            gap = self.np_random.uniform(base_gap * 0.8, base_gap * 1.2) * gap_multiplier
            current_x += gap
            
            width_multiplier = 1.0 - 0.08 * (stage_num - 1)
            platform_width = self.np_random.uniform(base_platform_width * 0.7, base_platform_width * 1.3) * width_multiplier
            platform_width = max(self.PLAYER_SIZE * 2, platform_width)

            # Ensure verticality is not too extreme
            last_y = self.platforms[-1].y
            y_change = self.np_random.uniform(-40, 40)
            platform_y = np.clip(last_y + y_change, 200, 350)

            platform_rect = pygame.Rect(current_x, platform_y, platform_width, self.screen_height - platform_y)
            self.platforms.append(platform_rect)
            current_x += platform_width
        
        # End flag
        flag_x = current_x + base_gap * 2
        flag_y = self.platforms[-1].y - 50
        self.end_flag = pygame.Rect(flag_x, flag_y, 15, 50)


    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        self.stage_timer += 1
        
        # --- Action Handling ---
        # Movement (actions[0]) and shift (actions[2]) have no effect.
        space_pressed = action[1] == 1

        # --- Game Logic ---
        # 1. Player Horizontal Movement (automatic)
        self.player_pos[0] += self.PLAYER_SPEED
        reward += 0.1 # Reward for forward progress

        # 2. Player Vertical Movement (Jump & Gravity)
        if space_pressed and self.player_on_ground:
            self.player_vel_y = self.JUMP_STRENGTH
            self.player_on_ground = False
            self.just_landed_reward_given = False
            # sfx: jump
        elif not space_pressed and self.player_on_ground:
            reward -= 0.01 # Penalty for inaction

        self.player_vel_y += self.GRAVITY
        self.player_pos[1] += self.player_vel_y
        
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)

        # 3. Collision Detection
        self.player_on_ground = False
        for plat in self.platforms:
            if player_rect.colliderect(plat):
                # FIX: The original condition for landing was too strict and susceptible to
                # float-to-int conversion errors, causing the player to fall through platforms.
                # This simpler condition robustly handles landings by checking if the player is
                # moving downwards and has collided with a platform.
                if self.player_vel_y >= 0:
                    self.player_pos[1] = plat.top - self.PLAYER_SIZE
                    self.player_vel_y = 0
                    self.player_on_ground = True
                    if not self.just_landed_reward_given:
                        reward += 1.0 # Reward for landing
                        self.just_landed_reward_given = True
                        self._create_landing_particles(player_rect.midbottom)
                        # sfx: land
                    break # Stop checking other platforms

        # 4. Particle Update
        self._update_particles()
        
        # 5. Termination & Progression
        # Pit Fall
        if self.player_pos[1] > self.screen_height:
            self.terminated = True
            reward = -10
            self.death_flash_alpha = 200
            # sfx: fall_death
        
        # Max steps
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.terminated = True

        # Stage/Game Complete
        if self.end_flag and player_rect.colliderect(self.end_flag):
            if self.stage < self.NUM_STAGES:
                reward += 10
                self.stage += 1
                self._generate_stage(self.stage)
                self.player_pos = [self.player_pos[0] + 150, 100] # Move forward, give vertical clearance
                self.player_vel_y = 0
                self.stage_timer = 0
                self.just_landed_reward_given = True # Reset landing state
                # sfx: stage_complete
            else:
                reward += 50
                self.terminated = True
                # sfx: game_win

        self.score += reward
        
        # In Gymnasium, `terminated` should be returned if the episode ends due to a game-over condition,
        # and `truncated` if it ends due to a time limit or other external factor.
        # Here, we combine them for simplicity in the internal state `self.terminated`.
        # The return value distinguishes between them.
        is_terminated_by_game_rules = self.terminated and not truncated
        
        return (
            self._get_observation(),
            reward,
            is_terminated_by_game_rules,
            truncated,
            self._get_info()
        )

    def _create_landing_particles(self, pos):
        for _ in range(10):
            vel_x = self.np_random.uniform(-1.5, 1.5)
            vel_y = self.np_random.uniform(-2, -0.5)
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([list(pos), [vel_x, vel_y], lifetime])

    def _update_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[1][1] += 0.1 # particle gravity
            p[2] -= 1
        self.particles = [p for p in self.particles if p[2] > 0]
        
    def _get_observation(self):
        # Update camera to follow player with a lead
        target_camera_x = self.player_pos[0] - self.screen_width * 0.3
        # Smooth camera movement
        self.camera_offset_x += (target_camera_x - self.camera_offset_x) * 0.1

        # --- Rendering ---
        self.screen.fill(self.COLOR_BG)
        
        self._render_game()
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Platforms
        for plat in self.platforms:
            # Simple culling
            if plat.right > self.camera_offset_x and plat.left < self.camera_offset_x + self.screen_width:
                render_rect = plat.move(-self.camera_offset_x, 0)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, render_rect)

        # Render End Flag
        if self.end_flag:
            flag_rect_cam = self.end_flag.move(-self.camera_offset_x, 0)
            pygame.draw.rect(self.screen, self.COLOR_GOAL, flag_rect_cam)
            pygame.draw.line(self.screen, self.COLOR_UI, 
                             (flag_rect_cam.left, flag_rect_cam.top), 
                             (flag_rect_cam.left, flag_rect_cam.bottom + 50), 2)

        # Render Particles
        for p in self.particles:
            pos_x, pos_y = p[0]
            size = max(0, p[2] / 6)
            pygame.draw.circle(self.screen, self.COLOR_PARTICLE, 
                               (int(pos_x - self.camera_offset_x), int(pos_y)), 
                               int(size))
        
        # Render Player
        player_screen_x = int(self.player_pos[0] - self.camera_offset_x)
        player_screen_y = int(self.player_pos[1])
        player_rect_on_screen = pygame.Rect(player_screen_x, player_screen_y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect_on_screen)
        
        # Render death flash
        if self.death_flash_alpha > 0:
            flash_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            flash_surface.fill((*self.COLOR_DANGER, self.death_flash_alpha))
            self.screen.blit(flash_surface, (0, 0))
            self.death_flash_alpha = max(0, self.death_flash_alpha - 10)

    def _render_ui(self):
        stage_text = self.font_ui.render(f"Stage: {self.stage}/{self.NUM_STAGES}", True, self.COLOR_UI)
        self.screen.blit(stage_text, (10, 10))

        time_seconds = self.stage_timer / 30.0 # Assuming 30 FPS
        timer_text = self.font_ui.render(f"Time: {time_seconds:.2f}", True, self.COLOR_UI)
        self.screen.blit(timer_text, (self.screen_width - timer_text.get_width() - 10, 10))
        
        if self.terminated and self.player_pos[1] <= self.screen_height: # If won
            win_text = self.font_game_over.render("YOU WIN!", True, self.COLOR_GOAL)
            text_rect = win_text.get_rect(center=(self.screen_width/2, self.screen_height/2))
            self.screen.blit(win_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "player_x": self.player_pos[0],
            "player_y": self.player_pos[1],
        }
        
    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # To run with rendering, comment out the os.environ line in __init__
    # and set render_mode="human"
    env = GameEnv()
    
    # --- To run the game with manual controls ---
    # For human rendering, you'll need a display.
    # If you get an error, ensure you have a display server running (e.g., X11, Wayland)
    # or run in a virtual framebuffer like xvfb.
    try:
        pygame.display.set_caption(env.game_description)
        screen = pygame.display.set_mode((env.screen_width, env.screen_height))
        human_render = True
    except pygame.error:
        print("Pygame display could not be initialized. Running headlessly.")
        human_render = False

    obs, info = env.reset()
    terminated = False
    truncated = False
    
    # Game loop
    running = True
    while running:
        action = env.action_space.sample() 
        action.fill(0) # Start with a no-op action

        if human_render:
            # Action mapping from keyboard
            keys = pygame.key.get_pressed()
            action[1] = 1 if keys[pygame.K_SPACE] else 0 # Space to jump
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r: # Reset on 'r'
                        obs, info = env.reset()
                        terminated = False
                        truncated = False
        
        if not terminated and not truncated:
            # The game is an auto-runner, so we always want to move forward.
            # Let's map action[0] = 1 to 'move right'.
            action[0] = 1
            obs, reward, terminated, truncated, info = env.step(action)
        
        if human_render:
            # Display the observation from the environment
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(30) # Control FPS
        else: # Headless mode, just run a few steps
            if env.steps > 200:
                running = False


    env.close()