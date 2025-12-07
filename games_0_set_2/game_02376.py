
# Generated: 2025-08-27T20:11:05.849851
# Source Brief: brief_02376.md
# Brief Index: 2376

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the harvester. Hold Space to activate the collection beam."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a harvester ship to collect falling alien resources. Green, blue, and purple "
        "resources offer increasing rewards. Avoid the pulsating red hazard zones. Collect 50 "
        "resources before the 60-second timer runs out to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds
        self.WIN_RESOURCES = 50

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_score = pygame.font.Font(None, 48)
        self.font_game_over = pygame.font.Font(None, 72)

        # --- Colors ---
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_BEAM = (220, 220, 255)
        self.COLOR_RES_GREEN = (50, 255, 100)
        self.COLOR_RES_BLUE = (100, 150, 255)
        self.COLOR_RES_PURPLE = (220, 100, 255)
        self.COLOR_HAZARD = (255, 50, 50)
        self.COLOR_TEXT = (240, 240, 240)
        
        # --- Game State Attributes ---
        self.player_x = 0
        self.player_width = 70
        self.player_height = 15
        self.player_speed = 10
        self.beam_active = False
        self.beam_width = 100
        self.beam_height = 180

        self.resources = []
        self.hazards = []
        self.particles = []
        self.stars = self._generate_stars()
        
        self.steps = 0
        self.score = 0.0
        self.collected_resources = 0
        self.game_over = False
        self.game_won = False
        self.reward_this_step = 0.0
        
        self.hazard_spawn_timer = 0
        self.hazard_level_up_interval = 15 * self.FPS
        self.hazard_spawn_prob = 0.01
        self.hazard_max_radius = 40

        # Initialize state and validate
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_x = self.SCREEN_WIDTH // 2
        self.beam_active = False
        
        self.resources = []
        self.hazards = []
        self.particles = []
        
        self.steps = 0
        self.score = 0.0
        self.collected_resources = 0
        self.game_over = False
        self.game_won = False
        
        self.hazard_spawn_prob = 0.01
        self.hazard_max_radius = 40
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_this_step = -0.2  # Base penalty for each step not collecting

        self._handle_input(action)
        self._update_game_state()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            self.game_over = True
            if self.game_won:
                self.reward_this_step += 50
            else: # Timeout
                self.reward_this_step -= 100
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        if movement == 3:  # Left
            self.player_x -= self.player_speed
        elif movement == 4:  # Right
            self.player_x += self.player_speed
        
        self.player_x = np.clip(self.player_x, self.player_width // 2, self.SCREEN_WIDTH - self.player_width // 2)
        self.beam_active = space_held

    def _update_game_state(self):
        self._update_hazards()
        self._update_resources()
        self._update_particles()
        self._handle_collisions()

    def _update_hazards(self):
        # Level up hazard difficulty
        if self.steps > 0 and self.steps % self.hazard_level_up_interval == 0:
            self.hazard_spawn_prob *= 1.10
            self.hazard_max_radius *= 1.05

        # Spawn new hazards
        if self.np_random.random() < self.hazard_spawn_prob:
            radius = self.np_random.integers(20, int(self.hazard_max_radius))
            x = self.np_random.integers(radius, self.SCREEN_WIDTH - radius)
            y = self.np_random.integers(radius, self.SCREEN_HEIGHT - 100) # Keep away from player start
            lifetime = self.np_random.integers(5 * self.FPS, 10 * self.FPS)
            self.hazards.append({'pos': pygame.Vector2(x, y), 'radius': radius, 'max_radius': radius, 'lifetime': lifetime, 'spawn_step': self.steps})

        # Update existing hazards
        self.hazards = [h for h in self.hazards if h['lifetime'] > 0]
        for h in self.hazards:
            h['lifetime'] -= 1
            # Pulsating effect
            pulse = math.sin((self.steps - h['spawn_step']) * 0.2) * 0.2 + 0.8
            h['radius'] = h['max_radius'] * pulse

    def _update_resources(self):
        # Spawn new resources
        spawn_prob = 0.03 + 0.05 * (self.steps / self.MAX_STEPS) # Rate increases over time
        if self.np_random.random() < spawn_prob:
            x = self.np_random.integers(10, self.SCREEN_WIDTH - 10)
            speed = self.np_random.uniform(1.5, 3.5)
            
            type_roll = self.np_random.random()
            if type_roll < 0.6: # 60% chance
                res_type = 'green'
                value = 1
                reward = 1
                color = self.COLOR_RES_GREEN
            elif type_roll < 0.9: # 30% chance
                res_type = 'blue'
                value = 3
                reward = 3
                color = self.COLOR_RES_BLUE
            else: # 10% chance
                res_type = 'purple'
                value = 5
                reward = 5
                color = self.COLOR_RES_PURPLE
            
            self.resources.append({'pos': pygame.Vector2(x, -10), 'speed': speed, 'type': res_type, 'value': value, 'reward': reward, 'color': color})

        # Move and remove off-screen resources
        for res in self.resources:
            res['pos'].y += res['speed']
        self.resources = [res for res in self.resources if res['pos'].y < self.SCREEN_HEIGHT + 20]

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'].y += 0.1 # Gravity
            p['lifetime'] -= 1

    def _handle_collisions(self):
        # Resource collection
        if self.beam_active:
            beam_rect = pygame.Rect(
                self.player_x - self.beam_width // 2, 
                self.SCREEN_HEIGHT - self.player_height - self.beam_height, 
                self.beam_width, 
                self.beam_height
            )
            
            collected_this_frame = []
            for res in self.resources:
                if beam_rect.collidepoint(res['pos'].x, res['pos'].y):
                    collected_this_frame.append(res)
                    self.score += res['value']
                    self.reward_this_step += res['reward']
                    self.collected_resources += 1
                    self._create_particles(res['pos'], res['color'], 15)
                    # sfx: collection sound based on res['type']
            
            self.resources = [res for res in self.resources if res not in collected_this_frame]

        # Hazard collision
        player_rect = pygame.Rect(self.player_x - self.player_width // 2, self.SCREEN_HEIGHT - self.player_height, self.player_width, self.player_height)
        for h in self.hazards:
            dist_x = abs(h['pos'].x - player_rect.centerx)
            dist_y = abs(h['pos'].y - player_rect.centery)
            
            if dist_x < (player_rect.width / 2 + h['radius']) and dist_y < (player_rect.height / 2 + h['radius']):
                self.reward_this_step -= 1.0 # Penalty for being in a hazard
                self.score -= 0.1
                if self.np_random.random() < 0.2:
                    self._create_particles(pygame.Vector2(player_rect.centerx, player_rect.top), self.COLOR_HAZARD, 3, 'spark')
                # sfx: player damage sizzle

    def _check_termination(self):
        if self.collected_resources >= self.WIN_RESOURCES:
            self.game_won = True
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_stars()
        self._render_hazards()
        self._render_resources()
        self._render_particles()
        self._render_player()
        self._render_ui()

        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "collected_resources": self.collected_resources,
            "time_remaining": (self.MAX_STEPS - self.steps) / self.FPS
        }

    # --- Rendering Methods ---

    def _render_stars(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, star['color'], (star['x'], star['y']), star['size'])

    def _render_player(self):
        player_y = self.SCREEN_HEIGHT - self.player_height
        player_rect = (
            int(self.player_x - self.player_width / 2),
            int(player_y),
            self.player_width,
            self.player_height
        )
        
        # Ship body
        ship_points = [
            (player_rect[0], player_rect[1] + self.player_height),
            (player_rect[0] + self.player_width / 2, player_rect[1]),
            (player_rect[0] + self.player_width, player_rect[1] + self.player_height)
        ]
        pygame.gfxdraw.aapolygon(self.screen, ship_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, ship_points, self.COLOR_PLAYER)

        # Collection Beam
        if self.beam_active:
            beam_y_top = player_y - self.beam_height
            beam_points = [
                (self.player_x - self.beam_width / 2, player_y),
                (self.player_x + self.beam_width / 2, player_y),
                (self.player_x + self.beam_width / 4, beam_y_top),
                (self.player_x - self.beam_width / 4, beam_y_top)
            ]
            
            # Main beam with low alpha
            alpha = 80 + math.sin(self.steps * 0.5) * 20
            pygame.gfxdraw.filled_polygon(self.screen, beam_points, (*self.COLOR_BEAM, int(alpha)))
            
            # Animated energy lines
            for i in range(4):
                line_alpha = 100 + math.sin(self.steps * 0.3 + i) * 50
                y_offset = (self.steps * 4 + i * (self.beam_height/4)) % self.beam_height
                start_y = player_y - y_offset
                if start_y > beam_y_top:
                    pygame.draw.line(self.screen, (*self.COLOR_BEAM, line_alpha), (beam_points[0][0], start_y), (beam_points[3][0], start_y), 1)
                    pygame.draw.line(self.screen, (*self.COLOR_BEAM, line_alpha), (beam_points[1][0], start_y), (beam_points[2][0], start_y), 1)

    def _render_resources(self):
        for res in self.resources:
            x, y = int(res['pos'].x), int(res['pos'].y)
            color = res['color']
            
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, x, y, 10, (*color, 50))
            
            if res['type'] == 'green':
                pygame.gfxdraw.aacircle(self.screen, x, y, 7, color)
                pygame.gfxdraw.filled_circle(self.screen, x, y, 7, color)
            elif res['type'] == 'blue':
                size = 14
                rect = pygame.Rect(x - size/2, y - size/2, size, size)
                pygame.draw.rect(self.screen, color, rect)
            elif res['type'] == 'purple':
                size = 10
                points = [(x, y-size), (x+size, y), (x, y+size), (x-size, y)]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_hazards(self):
        for h in self.hazards:
            x, y = int(h['pos'].x), int(h['pos'].y)
            radius = int(h['radius'])
            alpha = min(255, h['lifetime'] * 2, (h['spawn_step'] + h['max_radius'] * 5 - self.steps) * 2) # Fade in/out
            alpha = max(0, alpha)
            
            # Pulsating glow
            pulse_alpha = (math.sin(self.steps * 0.2) * 0.5 + 0.5) * 100 * (alpha / 255)
            pygame.gfxdraw.filled_circle(self.screen, x, y, int(radius * 1.2), (*self.COLOR_HAZARD, int(pulse_alpha)))
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, (*self.COLOR_HAZARD, alpha))

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p['lifetime'] / p['max_lifetime']
            alpha = int(255 * life_ratio)
            color = (*p['color'], alpha)
            pos = (int(p['pos'].x), int(p['pos'].y))
            if p['type'] == 'spark':
                size = int(p['size'] * life_ratio)
                pygame.draw.line(self.screen, color, pos, (pos[0] + p['vel'].x, pos[1] + p['vel'].y), max(1, size))
            else:
                size = int(p['size'] * life_ratio)
                pygame.draw.circle(self.screen, color, pos, max(1, size))

    def _render_ui(self):
        # Resource Count
        res_text = f"RESOURCES: {self.collected_resources}/{self.WIN_RESOURCES}"
        res_surf = self.font_ui.render(res_text, True, self.COLOR_TEXT)
        self.screen.blit(res_surf, (10, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = f"TIME: {time_left:.1f}"
        timer_color = self.COLOR_HAZARD if time_left < 10 else self.COLOR_TEXT
        timer_surf = self.font_ui.render(timer_text, True, timer_color)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 10, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_score.render(score_text, True, self.COLOR_TEXT)
        score_pos = (self.SCREEN_WIDTH // 2 - score_surf.get_width() // 2, self.SCREEN_HEIGHT - 50)
        self.screen.blit(score_surf, score_pos)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        text = "VICTORY" if self.game_won else "TIME UP"
        color = self.COLOR_PLAYER if self.game_won else self.COLOR_HAZARD
        
        text_surf = self.font_game_over.render(text, True, color)
        text_pos = (self.SCREEN_WIDTH // 2 - text_surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - text_surf.get_height() // 2)
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_pos)

    # --- Helper Utilities ---

    def _generate_stars(self):
        stars = []
        for _ in range(150):
            x = random.randint(0, self.SCREEN_WIDTH)
            y = random.randint(0, self.SCREEN_HEIGHT)
            size = random.choice([1, 1, 1, 2])
            color_val = random.randint(100, 200)
            color = (color_val, color_val, color_val)
            stars.append({'x': x, 'y': y, 'size': size, 'color': color})
        return stars
    
    def _create_particles(self, pos, color, count, p_type='default'):
        for _ in range(count):
            if p_type == 'spark':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 6)
                vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
                lifetime = self.np_random.integers(10, 20)
                size = self.np_random.integers(2, 4)
            else: # Default collection particle
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
                lifetime = self.np_random.integers(20, 40)
                size = self.np_random.integers(2, 5)

            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'lifetime': lifetime, 'max_lifetime': lifetime,
                'color': color, 'size': size, 'type': p_type
            })

    def close(self):
        pygame.font.quit()
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Set SDL_VIDEODRIVER to "dummy" for headless execution
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Harvester Havoc")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to get it onto the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.0f}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        # --- Frame Rate Control ---
        clock.tick(env.FPS)
        
    env.close()